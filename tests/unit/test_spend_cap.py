"""A provider's spend cap is MONEY (amount + currency) declared once in the machine file; tokens are derived."""

import json
import math

import pytest

from ai_lab import ResponseRequest, SpendCapReached, Text, Unsupported, budget, connect
from ai_lab.prices import Price, Quote

_KEY = budget.fingerprint('ak-test')

_REQUEST = ResponseRequest('i', (Text('t'),), {'type': 'object', 'properties': {}})

#: One US dollar per million input tokens, two per million output: 1000 in + 500 out cost 0.002 USD.
_RATE = Quote(Price(1.0, 2.0), 'test', agreed=True, sources={})


def _reply(inp, out):
    def post(call, **_):
        post.calls.append(call)
        return {'content': [{'type': 'text', 'text': '{}'}], 'usage': {'input_tokens': inp, 'output_tokens': out}}

    post.calls = []
    return post


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """``machine(text)``: the per-machine file is exactly *text*; every model is priced at :data:`_RATE`."""

    def write(text):
        path = tmp_path / 'machine.toml'
        path.write_text(text, encoding='utf-8')
        monkeypatch.setenv('LAB_COMMONS_CONFIG', str(path))

    monkeypatch.setattr(budget, 'rate', lambda client, at=None: _RATE)
    return write


def _caps(**rows):
    return '[ai_lab.spend_cap]\n' + ''.join(f'"{k}" = {v}\n' for k, v in rows.items())


def _usd(amount):
    return f'{{ amount = {amount}, currency = "USD" }}'


def test_a_paid_provider_with_no_declared_cap_refuses_naming_where_to_declare_it(keys, machine):
    machine('')
    post = _reply(1, 1)
    with pytest.raises(Unsupported, match=r'\[ai_lab\.spend_cap\] anthropic'):
        connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert post.calls == []


def test_an_explicitly_uncapped_provider_calls_without_a_price(keys, machine, monkeypatch):
    machine(_caps(anthropic=_usd('inf')))
    monkeypatch.setattr(budget, 'rate', lambda client, at=None: None)
    post = _reply(1, 1)
    connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert budget.left_usd('anthropic') == math.inf
    assert post.calls[0].body['max_tokens'] == 8192


def test_a_keyless_provider_pays_nothing_and_needs_no_cap(machine):
    machine('')
    assert budget.left_usd('ollama') == math.inf


def test_a_call_charges_its_dollars_to_its_provider_and_its_key(keys, machine):
    machine(_caps(anthropic=_usd(1)))
    connect('anthropic:claude-x', post=_reply(1000, 500)).respond(_REQUEST)
    assert budget.spent_usd('anthropic') == pytest.approx(0.002)
    assert budget.spent_usd('anthropic', _KEY) == pytest.approx(0.002)
    assert budget.left_usd('anthropic') == pytest.approx(0.998)


def test_planted_control_a_roomy_cap_leaves_the_wire_ceiling(keys, machine):
    machine(_caps(anthropic=_usd(50)))
    post = _reply(1, 1)
    connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert post.calls[0].body['max_tokens'] == 8192


def test_max_tokens_is_what_the_tightest_scope_has_left_at_the_output_rate(keys, machine):
    machine(_caps(**{'anthropic': _usd(10), f'anthropic/{_KEY}': _usd(0.004)}))
    connect('anthropic:claude-x', post=_reply(1000, 500)).respond(_REQUEST)  # 0.002 USD
    post = _reply(1, 1)
    connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert post.calls[0].body['max_tokens'] == 1000  # 0.002 USD left at 2 USD per million output tokens


def test_a_provider_at_its_cap_refuses_by_scope_and_sends_nothing(keys, machine):
    machine(_caps(anthropic=_usd(0.002)))
    connect('anthropic:claude-x', post=_reply(1000, 500)).respond(_REQUEST)
    post = _reply(1, 1)
    with pytest.raises(SpendCapReached, match=r'anthropic has spent 0\.0020 of its 0\.0020 USD') as spent:
        connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert spent.value.scope == 'anthropic'
    assert post.calls == []


def test_a_key_at_its_own_cap_is_refused_while_its_provider_has_room(keys, machine):
    machine(_caps(**{'anthropic': _usd(10), f'anthropic/{_KEY}': _usd(0.002)}))
    connect('anthropic:claude-x', post=_reply(1000, 500)).respond(_REQUEST)
    post = _reply(1, 1)
    with pytest.raises(SpendCapReached, match=rf'anthropic/{_KEY} spend cap') as spent:
        connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert spent.value.scope == f'anthropic/{_KEY}'
    assert post.calls == []


def test_one_providers_spend_never_draws_on_another(keys, machine):
    machine(_caps(anthropic=_usd(1), openai=_usd(1)))
    connect('anthropic:claude-x', post=_reply(1000, 500)).respond(_REQUEST)
    assert budget.left_usd('openai') == 1


def test_a_cap_in_another_currency_converts_at_the_declared_rate(machine):
    machine(_caps(deepseek='{ amount = 100, currency = "CNY" }') + '[ai_lab.usd_per]\nCNY = 0.14\n')
    assert budget.left_usd('deepseek') == pytest.approx(14.0)
    assert budget.cap('deepseek') == budget.Cap(100, 'CNY')


def test_a_currency_with_no_declared_rate_refuses(machine):
    machine(_caps(deepseek='{ amount = 100, currency = "CNY" }'))
    with pytest.raises(Unsupported, match=r'\[ai_lab\.usd_per\] CNY'):
        budget.left_usd('deepseek')


def test_a_capped_call_whose_model_has_no_price_refuses(keys, machine, monkeypatch):
    machine(_caps(anthropic=_usd(1)))
    monkeypatch.setattr(budget, 'rate', lambda client, at=None: None)
    post = _reply(1, 1)
    with pytest.raises(Unsupported, match=r'anthropic:claude-x has no price'):
        connect('anthropic:claude-x', post=post).respond(_REQUEST)
    assert post.calls == []


@pytest.mark.parametrize('row', ['1000', '{ amount = -1, currency = "USD" }', '{ amount = 1 }'])
def test_a_malformed_cap_is_refused(machine, row):
    machine(_caps(anthropic=row))
    with pytest.raises(ValueError, match=r'ai_lab\.spend_cap'):
        budget.left_usd('anthropic')


def test_the_ledger_names_provider_and_key_fingerprint_and_never_a_secret(keys, machine):
    machine(_caps(anthropic=_usd(1)))
    connect('anthropic:claude-x', post=_reply(7, 3)).respond(_REQUEST)
    text = budget.ledger_file().read_text('utf-8')
    (row,) = [json.loads(line) for line in text.splitlines()]
    assert {k: row[k] for k in ('provider', 'key', 'tokens')} == {'provider': 'anthropic', 'key': _KEY, 'tokens': 10}
    assert 'ak-test' not in text
