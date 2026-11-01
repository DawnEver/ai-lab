"""Daily token budgets per provider and per API key: a NUMBER of tokens, metered from each reply's usage."""

import json

import pytest

from ai_lab import DailyTokenCapReached, ResponseRequest, Text, budget, connect

_KEY = budget.fingerprint('ak-test')

_REQUEST = ResponseRequest('i', (Text('t'),), {'type': 'object', 'properties': {}})


def _reply(inp, out):
    def post(call, **_):
        post.calls.append(call)
        return {'content': [{'type': 'text', 'text': '{}'}], 'usage': {'input_tokens': inp, 'output_tokens': out}}

    post.calls = []
    return post


@pytest.fixture
def caps(tmp_path, monkeypatch):
    def set_caps(**table):
        path = tmp_path / 'machine.toml'
        rows = ''.join(f'"{k}" = {v}\n' for k, v in table.items())
        path.write_text(f'[ai_lab.daily_token_cap]\n{rows}', encoding='utf-8')
        monkeypatch.setenv('LAB_COMMONS_CONFIG', str(path))

    return set_caps


def test_the_default_cap_holds_for_a_provider_the_machine_file_does_not_name(keys, tmp_path, monkeypatch):
    monkeypatch.setenv('LAB_COMMONS_CONFIG', str(tmp_path / 'absent.toml'))
    assert budget.remaining('anthropic', _KEY) == budget.DEFAULT_DAILY_TOKEN_CAP


def test_a_call_charges_its_provider_and_its_key(keys, caps):
    caps(anthropic=1000)
    connect('anthropic:claude-x', post=_reply(7, 3)).respond(_REQUEST)
    connect('anthropic:claude-x', post=_reply(5, 5)).respond(_REQUEST)
    assert budget.used_today('anthropic') == 20
    assert budget.used_today('anthropic', _KEY) == 20
    assert budget.used_today('openai') == 0


def test_planted_control_under_the_cap_the_call_goes_out_with_the_wire_ceiling(keys, caps):
    caps(anthropic=100_000)
    post = _reply(1, 1)
    connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert post.calls[0].body['max_tokens'] == 8192


def test_max_tokens_never_asks_for_more_than_the_tightest_scope_has_left(keys, caps):
    caps(**{'anthropic': 10_000, f'anthropic/{_KEY}': 1000})
    connect('anthropic:claude-x', post=_reply(600, 100)).respond(_REQUEST)
    post = _reply(1, 1)
    connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert post.calls[0].body['max_tokens'] == 300


def test_a_provider_over_its_cap_refuses_by_scope_and_sends_nothing(keys, caps):
    caps(anthropic=10)
    connect('anthropic:claude-x', post=_reply(8, 2)).respond(_REQUEST)
    post = _reply(1, 1)
    with pytest.raises(DailyTokenCapReached, match=r'anthropic has used 10 of 10 tokens') as spent:
        connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert spent.value.scope == 'anthropic'
    assert post.calls == []


def test_one_providers_spend_never_draws_on_another(keys, caps):
    caps(anthropic=10, default=10)
    connect('anthropic:claude-x', post=_reply(8, 2)).respond(_REQUEST)
    assert budget.remaining('openai') == 10


def test_a_key_over_its_own_cap_is_refused_while_its_provider_has_room(keys, caps):
    caps(**{'anthropic': 10_000, f'anthropic/{_KEY}': 10})
    connect('anthropic:claude-x', post=_reply(8, 2)).respond(_REQUEST)
    post = _reply(1, 1)
    with pytest.raises(DailyTokenCapReached, match=rf'anthropic/{_KEY} daily token cap') as spent:
        connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert spent.value.scope == f'anthropic/{_KEY}'
    assert post.calls == []


def test_the_ledger_names_provider_and_key_fingerprint_and_never_a_secret(keys, caps):
    caps(anthropic=1000)
    connect('anthropic:claude-x', post=_reply(7, 3)).respond(_REQUEST)
    text = budget.ledger_file().read_text('utf-8')
    (row,) = [json.loads(line) for line in text.splitlines()]
    assert {k: row[k] for k in ('provider', 'key', 'tokens')} == {'provider': 'anthropic', 'key': _KEY, 'tokens': 10}
    assert 'ak-test' not in text
