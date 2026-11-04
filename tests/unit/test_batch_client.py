"""The Message Batches path of a client: one batch of calls, its state, its results by custom id. Fakes only."""

import json

import pytest

from ai_lab import ProviderError, ResponseRequest, Text, Unsupported, budget, connect
from ai_lab.client import Ticket
from ai_lab.errors import BatchLapsed

_REQ = ResponseRequest('Be brief.', (Text('prefix'), Text('change')), {'type': 'object', 'properties': {}})


def _message(value, usage=None):
    return {
        'type': 'message',
        'content': [{'type': 'text', 'text': json.dumps(value)}],
        'usage': usage or {'input_tokens': 10, 'output_tokens': 2},
    }


def _created(batch_id='msgbatch_01abc', status='in_progress'):
    return {
        'id': batch_id, 'type': 'message_batch', 'processing_status': status,
        'expires_at': '2026-11-05T10:00:00Z', 'results_url': None,
    }  # fmt: skip


class _Fake:
    """Records every POST and GET; answers from the queues it is given."""

    def __init__(self, posts=(), gets=()):
        self.posts, self.gets, self.sent, self.read = list(posts), list(gets), [], []

    def post(self, call, **_):
        self.sent.append(call)
        reply = self.posts.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def get(self, url, headers, **_):
        self.read.append((url, dict(headers)))
        return self.gets.pop(0)


def _client(fake, **options):
    return connect('anthropic:claude-x', post=fake.post, get=fake.get, **options)


def test_submit_sends_one_batch_whose_params_are_the_synchronous_body(keys):
    fake = _Fake(posts=[_created()])
    ticket = _client(fake).submit({'a1': _REQ, 'b2': _REQ})
    call = fake.sent[0]
    assert call.url == 'https://api.anthropic.com/v1/messages/batches'
    assert [r['custom_id'] for r in call.body['requests']] == ['a1', 'b2']
    params = call.body['requests'][0]['params']
    assert params['cache_control'] == {'type': 'ephemeral'}  # prompt caching stays on inside a batch
    assert params['messages'][0]['content'][0]['cache_control'] == {'type': 'ephemeral'}
    assert ticket == Ticket(id='msgbatch_01abc', slot=1, expires_at='2026-11-05T10:00:00Z')


def test_a_custom_id_the_api_refuses_is_refused_before_any_call(keys):
    fake = _Fake()
    with pytest.raises(ValueError, match='custom_id'):
        _client(fake).submit({'has space': _REQ})
    assert fake.sent == []


def test_a_wire_without_batches_refuses_to_submit(keys):
    with pytest.raises(Unsupported, match='no batch endpoint'):
        connect('openai:gpt-6-luna').submit({'a': _REQ})


def test_submit_moves_to_the_next_key_slot_on_an_account_failure(keys):
    keys('api.anthropic.com', 'ak-second', username='api-key-2')
    fake = _Fake(posts=[ProviderError('HTTP 429', kind='rate_limit'), _created()])
    ticket = _client(fake).submit({'a': _REQ})
    assert ticket.slot == 2
    assert fake.sent[1].headers['x-api-key'] == 'ak-second'


def test_ended_reads_the_processing_status_with_the_ticket_slot_key(keys):
    keys('api.anthropic.com', 'ak-second', username='api-key-2')
    fake = _Fake(gets=[json.dumps(_created()), json.dumps(_created(status='ended'))])
    client = _client(fake)
    ticket = Ticket(id='msgbatch_01abc', slot=2, expires_at='2026-11-05T10:00:00Z')
    assert client.ended(ticket) is False
    assert client.ended(ticket) is True
    assert fake.read[0][0] == 'https://api.anthropic.com/v1/messages/batches/msgbatch_01abc'
    assert fake.read[0][1]['x-api-key'] == 'ak-second'


def test_a_tampered_batch_id_is_refused_before_it_reaches_a_url(keys):
    fake = _Fake()
    with pytest.raises(ValueError, match='batch id'):
        _client(fake).ended(Ticket(id='../../v1/x?y', slot=1, expires_at=''))
    assert fake.read == []


def _errored(kind, message):
    return {'type': 'errored', 'error': {'type': 'error', 'error': {'type': kind, 'message': message}}}


def test_results_map_every_result_type_by_custom_id(keys):
    lines = [
        {'custom_id': 'ok', 'result': {'type': 'succeeded', 'message': _message({'findings': []})}},
        {'custom_id': 'limit', 'result': _errored('rate_limit_error', 'slow down')},
        {'custom_id': 'bad', 'result': _errored('invalid_request_error', 'no')},
        {'custom_id': 'late', 'result': {'type': 'expired'}},
        {'custom_id': 'gone', 'result': {'type': 'canceled'}},
    ]
    fake = _Fake(gets=['\n'.join(json.dumps(line) for line in reversed(lines)) + '\n'])
    got = _client(fake).results(Ticket(id='msgbatch_01abc', slot=1, expires_at=''), charge=False)
    assert fake.read[0][0] == 'https://api.anthropic.com/v1/messages/batches/msgbatch_01abc/results'
    assert got['ok'].value == {'findings': []}
    assert got['ok'].usage.input_tokens == 10
    assert (got['limit'].kind, got['bad'].kind) == ('rate_limit', '')
    assert 'slow down' in str(got['limit'])
    assert isinstance(got['late'], BatchLapsed)
    assert (got['late'].lapsed, got['late'].kind) == ('expired', 'timeout')
    assert isinstance(got['gone'], BatchLapsed)
    assert got['gone'].lapsed == 'canceled'


def test_results_charge_the_ledger_only_when_asked_and_mark_the_row_batch(keys):
    usage = {'input_tokens': 7, 'output_tokens': 3}
    line = {'custom_id': 'ok', 'result': {'type': 'succeeded', 'message': _message({}, usage)}}
    fake = _Fake(gets=[json.dumps(line), json.dumps(line)])
    client = _client(fake, tag='rev')
    ticket = Ticket(id='msgbatch_01abc', slot=1, expires_at='')
    client.results(ticket, charge=False)
    assert not budget.ledger_file().exists()
    client.results(ticket, charge=True)
    rows = [json.loads(r) for r in budget.ledger_file().read_text('utf-8').splitlines()]
    assert len(rows) == 1
    assert rows[0]['tokens'] == 10
    assert rows[0]['batch'] is True


def test_cancel_posts_to_the_cancel_endpoint(keys):
    fake = _Fake(posts=[_created(status='canceling')])
    _client(fake).cancel(Ticket('msgbatch_01abc', 1, ''))
    assert fake.sent[0].url == 'https://api.anthropic.com/v1/messages/batches/msgbatch_01abc/cancel'
