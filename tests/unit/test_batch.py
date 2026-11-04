"""The resumable batch driver: a program replayed until every call it makes has a reply. Fakes only, no network."""

import contextlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from ai_lab import ProviderError, Unsupported
from ai_lab.answers import Response
from ai_lab.batch import PENDING_EXIT, Batched, Direct, NotYet, Pending, State, digest, runner
from ai_lab.client import BatchLapsed, Ticket

_T0 = datetime(2026, 11, 4, 9, 0, tzinfo=UTC)


def _request(text, provider='anthropic', tag='r'):
    return {
        'verb': 'respond', 'provider': provider, 'model': 'claude-x', 'task': 'pr-review', 'tag': tag,
        'instructions': 'i', 'context': [{'kind': 'text', 'text': text}], 'schema': {'type': 'object'},
    }  # fmt: skip


class _Clock:
    def __init__(self):
        self.now = _T0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += timedelta(seconds=seconds)


class _Vendor:
    """Every client of one fake vendor: batches it holds, and when each ends (polls until ended)."""

    def __init__(self, polls_to_end=1, answer=lambda text: {'echo': text}, lapse=frozenset()):
        self.batches, self.polls_to_end, self.answer, self.lapse = {}, polls_to_end, answer, lapse
        self.submitted, self.charged, self.canceled = [], [], []

    def connect(self, spec, **options):
        return _FakeClient(self, spec, options)


class _FakeClient:
    can_batch = True

    def __init__(self, vendor, spec, options):
        self.vendor, self.spec, self.options = vendor, spec, options

    def submit(self, requests):
        batch_id = f'msgbatch_{len(self.vendor.batches) + 1}'
        self.vendor.batches[batch_id] = {'requests': dict(requests), 'polls': 0}
        self.vendor.submitted.append((self.spec, self.options, sorted(requests)))
        return Ticket(batch_id, 1, (_T0 + timedelta(hours=24)).isoformat())

    def ended(self, ticket):
        held = self.vendor.batches[ticket.id]
        held['polls'] += 1
        return held['polls'] >= self.vendor.polls_to_end

    def results(self, ticket, *, charge):
        self.vendor.charged.append((ticket.id, charge))
        out = {}
        for cid, request in self.vendor.batches[ticket.id]['requests'].items():
            text = request.context[0].text
            if text in self.vendor.lapse:
                out[cid] = BatchLapsed('the batch request expired before it ran', lapsed='expired')
            else:
                out[cid] = Response(self.vendor.answer(text), 'anthropic', 'claude-x', 0.0, None, 1)
        return out

    def cancel(self, ticket):
        self.vendor.canceled.append(ticket.id)


def _sync(request):
    return {'schema_version': 2, 'ok': True, 'kind': '', 'result': {'sync': request['context'][0]['text']},
            'usage': None, 'key_slot': 1, 'error': ''}  # fmt: skip


def _batched(vendor, clock, state=None, *, wait_s=3600.0, on_expiry='sync', answer=_sync):
    return Batched(state or State.fresh(None), connect=vendor.connect, answer=answer, clock=clock,
                   sleep=clock.sleep, wait_s=wait_s, on_expiry=on_expiry)  # fmt: skip


def _two_reviewers(call):
    """A program shaped like a review: each unit catches NotYet so every unit's call joins one batch."""
    out, missing = {}, False
    for name in ('a', 'b'):
        try:
            out[name] = call(_request(name))['result']
        except NotYet:
            missing = True
    if missing:
        raise NotYet
    return out


def test_every_unit_of_a_pass_joins_one_batch_and_the_replay_reads_it():
    vendor, clock = _Vendor(), _Clock()
    got = _batched(vendor, clock).settle(_two_reviewers)
    assert got == {'a': {'echo': 'a'}, 'b': {'echo': 'b'}}
    assert len(vendor.submitted) == 1
    assert vendor.submitted[0][2] == sorted([digest(_request('a')), digest(_request('b'))])


def test_a_conversation_is_one_batch_per_turn():
    vendor, clock = _Vendor(), _Clock()

    def turns(call):
        first = call(_request('turn 1'))
        return call(_request(f'turn 2 after {first["result"]["echo"]}'))['result']

    assert _batched(vendor, clock).settle(turns) == {'echo': 'turn 2 after turn 1'}
    assert len(vendor.submitted) == 2


def test_a_budget_spent_waiting_is_pending_and_the_state_names_the_batch(tmp_path):
    vendor, clock = _Vendor(polls_to_end=10**6), _Clock()
    path = tmp_path / 'state.json'
    got = _batched(vendor, clock, State.load(path), wait_s=600.0).settle(_two_reviewers)
    assert isinstance(got, Pending)
    assert got.ids == ('msgbatch_1',)
    assert 'msgbatch_1' in got.line()
    assert clock.now - _T0 <= timedelta(seconds=600)
    saved = json.loads(path.read_text('utf-8'))
    assert [b['id'] for b in saved['batches']] == ['msgbatch_1']
    assert 'key' not in json.dumps(saved).lower().replace('task', '')  # no secret, ever


def test_a_resumed_run_polls_the_recorded_batch_and_never_resubmits(tmp_path):
    vendor, clock = _Vendor(polls_to_end=3), _Clock()
    path = tmp_path / 'state.json'
    assert isinstance(_batched(vendor, clock, State.load(path), wait_s=0.0).settle(_two_reviewers), Pending)
    got = _batched(vendor, clock, State.load(path)).settle(_two_reviewers)
    assert got == {'a': {'echo': 'a'}, 'b': {'echo': 'b'}}
    assert len(vendor.submitted) == 1, 'the second run resumed the first run batch: no double spend'


def test_the_results_are_charged_once_across_resumed_runs(tmp_path):
    vendor, clock = _Vendor(), _Clock()
    path = tmp_path / 'state.json'
    _batched(vendor, clock, State.load(path)).settle(_two_reviewers)
    _batched(vendor, clock, State.load(path)).settle(_two_reviewers)
    assert [charge for _id, charge in vendor.charged] == [True, False]


def test_wait_zero_submits_and_returns_pending_at_once():
    vendor, clock = _Vendor(polls_to_end=2), _Clock()
    got = _batched(vendor, clock, wait_s=0.0).settle(_two_reviewers)
    assert isinstance(got, Pending)
    assert clock.now == _T0


def test_the_poll_backs_off():
    vendor, clock = _Vendor(polls_to_end=4), _Clock()
    pauses = []
    batched = _batched(vendor, clock)
    batched.sleep = lambda s: (pauses.append(s), clock.sleep(s))
    batched.settle(_two_reviewers)
    assert pauses == sorted(pauses)
    assert pauses[0] < pauses[-1]


def test_an_expired_request_is_answered_synchronously_when_the_declaration_says_sync():
    vendor, clock = _Vendor(lapse={'b'}), _Clock()
    got = _batched(vendor, clock, on_expiry='sync').settle(_two_reviewers)
    assert got == {'a': {'echo': 'a'}, 'b': {'sync': 'b'}}


def test_an_expired_request_is_a_timeout_reply_when_the_declaration_says_fail():
    vendor, clock = _Vendor(lapse={'b'}), _Clock()

    def program(call):
        return call(_request('b'))

    got = _batched(vendor, clock, on_expiry='fail').settle(program)
    assert (got['ok'], got['kind']) == (False, 'timeout')
    assert 'expired' in got['error']


def test_a_batch_running_past_its_expiry_is_canceled_and_taken_as_expired():
    vendor, clock = _Vendor(polls_to_end=10**6), _Clock()
    got = _batched(vendor, clock, wait_s=48 * 3600.0, on_expiry='fail').settle(lambda call: call(_request('a')))
    assert got['kind'] == 'timeout'
    assert vendor.canceled == ['msgbatch_1']


def test_a_provider_without_batches_is_answered_at_once():
    vendor, clock = _Vendor(), _Clock()

    def connect(spec, **options):
        if spec.startswith('openai'):
            msg = 'no batch endpoint'
            raise Unsupported(msg)
        return vendor.connect(spec, **options)

    batched = Batched(State.fresh(None), connect=connect, answer=_sync, clock=clock, sleep=clock.sleep, wait_s=60.0,
                      on_expiry='sync')  # fmt: skip
    assert batched.settle(lambda call: call(_request('x', provider='openai'))['result']) == {'sync': 'x'}
    assert vendor.submitted == []


def test_an_errored_request_is_its_classified_failure():
    class _Errs(_Vendor):
        pass

    vendor, clock = _Errs(), _Clock()
    original = _FakeClient.results

    def results(self, ticket, *, charge):
        got = original(self, ticket, charge=charge)
        return {cid: ProviderError('batch request errored: rate_limit_error', kind='rate_limit') for cid in got}

    _FakeClient.results = results
    try:
        got = _batched(vendor, clock).settle(lambda call: call(_request('a')))
    finally:
        _FakeClient.results = original
    assert (got['ok'], got['kind']) == (False, 'rate_limit')


def test_requests_for_different_clients_go_to_different_batches():
    vendor, clock = _Vendor(), _Clock()

    def program(call):
        with contextlib.suppress(NotYet):
            call(_request('a', tag='one'))
        return call(_request('b', tag='two'))

    _batched(vendor, clock).settle(program)
    assert sorted(options['tag'] for _spec, options, _ids in vendor.submitted) == ['one', 'two']


def test_the_seeded_rng_is_the_same_across_runs_and_differs_by_label(tmp_path):
    path = tmp_path / 'state.json'
    first = State.load(path)
    first.save()
    again = State.load(path)
    assert first.rng('x').random() == again.rng('x').random()
    assert first.rng('x').random() != first.rng('y').random()


def test_direct_answers_at_once_and_is_never_pending():
    assert Direct(answer=_sync).settle(lambda call: call(_request('a'))['result']) == {'sync': 'a'}


def test_runner_picks_direct_or_batched_from_the_flags(tmp_path):
    assert isinstance(runner(batch=False, state=None, wait_s=1.0, on_expiry='sync'), Direct)
    assert isinstance(runner(batch=True, state=str(tmp_path / 's.json'), wait_s=1.0, on_expiry='sync'), Batched)


def test_a_tampered_state_file_is_refused(tmp_path):
    path = tmp_path / 'state.json'
    path.write_text('{"seed": 1, "batches": "x"}', encoding='utf-8')
    with pytest.raises(ValueError, match='batch state'):
        State.load(path)


def test_the_pending_exit_is_ex_tempfail():
    assert PENDING_EXIT == 75
