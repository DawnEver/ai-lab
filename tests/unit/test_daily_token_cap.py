"""The one budget: a daily NUMBER of tokens, metered from the usage each reply reports."""

import json

import pytest

from ai_lab import DailyTokenCapReached, ResponseRequest, Text, budget, connect

_REQUEST = ResponseRequest('i', (Text('t'),), {'type': 'object', 'properties': {}})


def _reply(inp, out):
    def post(call, **_):
        post.calls.append(call)
        return {'content': [{'type': 'text', 'text': '{}'}], 'usage': {'input_tokens': inp, 'output_tokens': out}}

    post.calls = []
    return post


@pytest.fixture
def cap(tmp_path, monkeypatch):
    def set_cap(tokens):
        path = tmp_path / 'machine.toml'
        path.write_text(f'[ai_lab]\ndaily_token_cap = {tokens}\n', encoding='utf-8')
        monkeypatch.setenv('LAB_COMMONS_CONFIG', str(path))

    return set_cap


def test_the_default_cap_is_declared_when_the_machine_file_says_nothing(keys, tmp_path, monkeypatch):
    monkeypatch.setenv('LAB_COMMONS_CONFIG', str(tmp_path / 'absent.toml'))
    assert budget.daily_token_cap() == budget.DEFAULT_DAILY_TOKEN_CAP


def test_a_call_adds_its_input_and_output_tokens_to_the_days_ledger(keys, cap):
    cap(1000)
    connect('anthropic:claude-x', post=_reply(7, 3)).respond(_REQUEST)
    connect('anthropic:claude-x', post=_reply(5, 5)).respond(_REQUEST)
    assert budget.used_today() == 20
    assert budget.ledger_file().is_relative_to(budget.state_dir())


def test_planted_control_under_the_cap_the_call_goes_out_with_the_wire_ceiling(keys, cap):
    cap(100_000)
    post = _reply(1, 1)
    connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert post.calls[0].body['max_tokens'] == 8192


def test_max_tokens_never_asks_for_more_than_the_day_has_left(keys, cap):
    cap(1000)
    connect('anthropic:claude-x', post=_reply(600, 100)).respond(_REQUEST)
    post = _reply(1, 1)
    connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert post.calls[0].body['max_tokens'] == 300


def test_a_call_after_the_cap_is_refused_by_name_and_never_sent(keys, cap):
    cap(10)
    connect('anthropic:claude-x', post=_reply(8, 2)).respond(_REQUEST)
    post = _reply(1, 1)
    with pytest.raises(DailyTokenCapReached, match=r'10 of 10 tokens.*daily_token_cap'):
        connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert post.calls == []


def test_the_ledger_is_one_number_per_line_and_never_a_secret(keys, cap):
    cap(1000)
    connect('anthropic:claude-x', post=_reply(7, 3)).respond(_REQUEST)
    text = budget.ledger_file().read_text('utf-8')
    assert [json.loads(line) for line in text.splitlines()] == [10]
    assert 'ak-test' not in text
