"""REVIEW SPEND (user ruling 2026-11-03): a small diff is read in one pass, its stable prefix is cached, the summary's
budget is sized and the ledger names the profile each call was spent on.

Measured 2026-11-03 (``review-tiers/motronics-studio.jsonl``): small-tier reviews (median diff 7.8 kB) averaged 185k
input tokens over 6 turns with half of it cached, cost $11.4 over 29 runs and found no HIGH. Each claim here runs the
real wire through :func:`ai_lab.call.answer` with a recording ``post``: no network, no model.
"""

import json

import pytest

from ai_lab import ReplayMiss, ResponseRequest, Text, budget, connect
from ai_lab.call import answer
from ai_lab.ledger import Ledger
from ai_lab.review.__main__ import main
from ai_lab.review.explore import SMALL_TURNS, TURNS, turns_for
from ai_lab.review.summary import SOURCE_TOKENS, SUMMARY_TOKENS, SummaryTruncated, change_summary, output_budget
from tests.review_principles import PRINCIPLES_FILE
from tests.unit.test_review_explore import PAD, SENTINEL, Scripted, _repo, _review

_ASKS = {'requests': {'show': ['src/a.py']}}


class Wire:
    """A recording ``post`` for the anthropic wire: every HTTP body it was sent, each answered with *result*."""

    def __init__(self, result, *, stop_reason='end_turn'):
        self.result, self.stop_reason, self.bodies = result, stop_reason, []

    def __call__(self, call, **_):
        self.bodies.append(call.body)
        usage = {'input_tokens': 100, 'output_tokens': 10, 'cache_read_input_tokens': 90}
        text = json.dumps(self.result)
        return {'content': [{'type': 'text', 'text': text}], 'usage': usage, 'stop_reason': self.stop_reason}


@pytest.fixture
def review_key(credential_store):
    credential_store('api.anthropic.com', 'ak-review', username='api-key-pr-review')
    return credential_store


def _through(wire):
    return lambda request: answer(request, post=wire)


def test_a_small_change_is_reviewed_in_at_most_two_turns(tmp_path, capsys):
    repo, range_ = _repo(tmp_path, {'src/a.py': f'{SENTINEL}\n'})
    call = Scripted(_ASKS)
    _review(repo, range_, call, capsys)
    assert len(call.requests) == SMALL_TURNS == 2
    assert SENTINEL in '\n'.join(p['text'] for p in call.requests[0]['context']), 'the whole diff is sent up front'
    assert 'You have 2 turns' in call.requests[0]['context'][0]['text']


def test_planted_control_a_medium_change_keeps_every_turn(tmp_path, capsys):
    repo, range_ = _repo(tmp_path, {'src/a.py': f'{SENTINEL}\n', **PAD})
    call = Scripted(_ASKS)
    _review(repo, range_, call, capsys)
    assert len(call.requests) == TURNS
    assert (turns_for('small'), turns_for('medium'), turns_for('huge')) == (SMALL_TURNS, TURNS, TURNS)


def test_the_stable_prefix_is_the_first_block_behind_a_breakpoint_and_every_profile_shares_it(
    tmp_path, capsys, review_key
):
    repo, range_ = _repo(tmp_path, {'src/a.py': f'{SENTINEL}\n'})
    wire = Wire(_ASKS)
    argv = ['--principles', str(PRINCIPLES_FILE), '--rules', '', '--range', range_, '--model', 'claude-x',
            '--repo', str(repo), '--lens', 'strict', '--no-summary']  # fmt: skip
    assert main(argv, call=_through(wire)) == 0
    doc = json.loads(capsys.readouterr().out)
    profiles = doc['profiles']
    assert len(profiles) > 1, 'a strict lens plans several profiles: the prefix is shared across them'
    assert len(wire.bodies) == SMALL_TURNS * len(profiles)
    firsts = {body['messages'][0]['content'][0]['text'] for body in wire.bodies}
    (prefix,) = firsts
    assert SENTINEL in prefix and 'Principles:' in prefix, 'the rules and the diff are the stable prefix'
    assert 'Scope:' not in prefix, "a profile's own scope is never inside the shared prefix"
    for body in wire.bodies:
        assert body['messages'][0]['content'][0]['cache_control'] == {'type': 'ephemeral'}
        assert body['messages'][0]['content'][1]['text'].count('Scope:') == 1


def test_a_recorded_review_call_replays_its_stable_prefix_and_a_changed_prefix_misses(tmp_path, keys):
    schema = {'type': 'object', 'properties': {}}
    prefix, own = 'Rules ... Git diff: +x', 'Scope: bugs'
    request = ResponseRequest('i', (Text(prefix), Text(own)), schema)
    wire = Wire({})
    ledger = tmp_path / 'review.jsonl'
    Ledger(ledger, 'record').wrap(connect('anthropic:claude-x', post=wire)).respond(request)
    (sent,) = wire.bodies
    assert sent['messages'][0]['content'][0] == {'type': 'text', 'text': prefix, 'cache_control': {'type': 'ephemeral'}}
    replay = Ledger(ledger, 'replay').wrap(connect('anthropic:claude-x', post=Wire({'never': 'sent'})))
    assert replay.respond(request).value == {}
    with pytest.raises(ReplayMiss):
        replay.respond(ResponseRequest('i', (Text(prefix + ' changed'), Text(own)), schema))


def test_each_review_call_is_charged_to_its_profile(tmp_path, capsys, review_key, monkeypatch):
    monkeypatch.setenv(budget.TAG_ENV, 'DawnEver/motronics-studio#5149')
    repo, range_ = _repo(tmp_path, {'src/a.py': f'{SENTINEL}\n'})
    argv = ['--principles', str(PRINCIPLES_FILE), '--rules', '', '--range', range_, '--model', 'claude-x',
            '--repo', str(repo), '--lens', 'strict', '--no-summary']  # fmt: skip
    assert main(argv, call=_through(Wire({'findings': []}))) == 0
    profiles = json.loads(capsys.readouterr().out)['profiles']
    rows = [json.loads(line) for line in budget.ledger_file().read_text('utf-8').splitlines()]
    assert sorted(row['tag'] for row in rows) == sorted(f'DawnEver/motronics-studio#5149/{p}' for p in profiles)


def test_planted_control_a_call_with_no_tag_env_is_charged_to_its_profile_alone(keys, monkeypatch):
    monkeypatch.delenv(budget.TAG_ENV, raising=False)
    answer({'verb': 'respond', 'provider': 'anthropic', 'model': 'claude-x', 'instructions': 'i', 'tag': 'security',
            'context': [{'kind': 'text', 'text': 't'}], 'schema': {'type': 'object'}}, post=Wire({}))  # fmt: skip
    (row,) = [json.loads(line) for line in budget.ledger_file().read_text('utf-8').splitlines()]
    assert row['tag'] == 'security'


def _intent(commits):
    return {'commits': [{'sha': f's{n}', 'subject': f'c{n}', 'body': ''} for n in range(commits)],
            'resolutions': [], 'merge_audit': [], 'memory': []}  # fmt: skip


def test_the_summary_budget_grows_with_what_it_summarizes_and_reaches_the_wire(review_key):
    assert output_budget(_intent(0)) == SUMMARY_TOKENS
    assert output_budget(_intent(30)) == SUMMARY_TOKENS + 30 * SOURCE_TOKENS
    wire = Wire({'changes': [], 'open_items': []})
    change_summary(_intent(30), call=_through(wire), provider='anthropic', model='claude-x')
    (body,) = wire.bodies
    assert body['max_tokens'] == SUMMARY_TOKENS + 30 * SOURCE_TOKENS
    assert json.loads(budget.ledger_file().read_text('utf-8'))['tag'] == 'summary'


def test_a_summary_cut_at_its_budget_fails_loudly(review_key):
    wire = Wire({'changes': []}, stop_reason='max_tokens')
    with pytest.raises(SummaryTruncated, match='stop_reason=max_tokens'):
        change_summary(_intent(3), call=_through(wire), provider='anthropic', model='claude-x')


def test_a_truncated_summary_exits_3_before_any_reviewer_is_paid(tmp_path, capsys, review_key):
    repo, range_ = _repo(tmp_path, {'src/a.py': f'{SENTINEL}\n'})
    wire = Wire({'changes': []}, stop_reason='max_tokens')
    argv = ['--principles', str(PRINCIPLES_FILE), '--rules', '', '--range', range_, '--model', 'claude-x',
            '--repo', str(repo), '--lens', 'light']  # fmt: skip
    assert main(argv, call=_through(wire)) == 3
    out = capsys.readouterr()
    assert out.out == ''
    assert 'max_tokens' in out.err
    assert len(wire.bodies) == 1, 'only the summary call was made'
