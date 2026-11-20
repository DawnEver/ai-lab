"""Hierarchical spend budgets: provider cap > standing / task allotment > item share > call (its max_tokens).

A call is refused when ANY enclosing level has no room, and the refusal names that level; a child allotment
larger than what its parent leaves is refused at OPEN; every call's ledger row carries its scope, so spend is
attributable, and a scope reopened by name (another process, an HPC item) reads its spent from the ledger.
"""

import json
import subprocess
import sys

import pytest

from ai_lab import ResponseRequest, SpendCapReached, Text, budget, connect
from ai_lab.errors import Unsupported
from ai_lab.prices import Price, Quote

_REQUEST = ResponseRequest('i', (Text('t'),), {'type': 'object', 'properties': {}})

#: One US dollar per million input tokens, two per million output: 1000 in + 500 out cost 0.002 USD.
_RATE = Quote(Price(1.0, 2.0), 'test', agreed=True, sources={})
_CALL_USD = 0.002


def _reply(inp=1000, out=500):
    def post(call, **_):
        post.calls.append(call)
        return {'content': [{'type': 'text', 'text': '{}'}], 'usage': {'input_tokens': inp, 'output_tokens': out}}

    post.calls = []
    return post


@pytest.fixture
def machine(tmp_path, monkeypatch, keys):
    """``machine(cap, standing='')``: anthropic is capped at *cap* USD a month; every model at :data:`_RATE`."""

    def write(cap, standing=''):
        path = tmp_path / 'machine.toml'
        text = f'[ai_lab.spend_cap]\nanthropic = {{ amount = {cap}, currency = "USD" }}\n'
        path.write_text(text + (f'[ai_lab.standing]\n{standing}' if standing else ''), encoding='utf-8')
        monkeypatch.setenv('LAB_COMMONS_CONFIG', str(path))

    monkeypatch.setattr(budget, 'rate', lambda client, at=None: _RATE)
    monkeypatch.delenv(budget.SCOPE_ENV, raising=False)
    return write


def _call(post=None):
    connect('anthropic:claude-x', post=post or _reply()).respond(_REQUEST)


def test_a_call_inside_a_task_is_refused_naming_the_task_when_the_task_is_spent(machine):
    machine(1)
    with budget.allot('campaign:T3', usd=_CALL_USD, provider='anthropic'):
        _call()
        post = _reply()
        with pytest.raises(SpendCapReached) as refused:
            _call(post)
    assert refused.value.scope == 'campaign:T3'
    assert post.calls == []


def test_a_call_inside_an_item_is_refused_naming_the_item_while_its_task_has_room(machine):
    machine(1)
    with budget.allot('campaign:T3', usd=0.5, provider='anthropic'), budget.allot('campaign:T3>item:1', usd=_CALL_USD):
        _call()
        with pytest.raises(SpendCapReached) as refused:
            _call()
    assert refused.value.scope == 'campaign:T3>item:1'


def test_a_call_inside_a_task_is_refused_naming_the_provider_when_the_month_is_spent(machine):
    machine(1)
    with budget.allot('campaign:T3', usd=0.5, provider='anthropic'):
        budget.charge('anthropic', None, budget.Usage(1_000_000, 0), 'claude-x')  # 1 USD outside any scope
        with pytest.raises(SpendCapReached) as refused:
            _call()
    assert refused.value.scope == 'anthropic'


def test_an_unscoped_call_cannot_spend_what_a_standing_budget_holds(machine):
    machine(0.003, standing='review = { provider = "anthropic", amount = 0.002, currency = "USD" }\n')
    _call()  # 0.003 cap - 0.002 held for review leaves 0.001: room, the call starts
    with pytest.raises(SpendCapReached) as refused:
        _call()
    assert refused.value.scope == 'anthropic'
    with budget.allot('review'):
        _call()  # review's own carve-out is its to spend


def test_the_tightest_level_derives_the_calls_max_tokens(machine):
    machine(50)
    post = _reply()
    with budget.allot('campaign:T3', usd=1, provider='anthropic'), budget.allot('campaign:T3>item:1', usd=0.0001):
        _call(post)
    assert post.calls[0].body['max_tokens'] == 50  # 0.0001 USD at 2 USD per million output tokens


def test_planted_control_a_child_larger_than_its_parent_leaves_is_refused_at_open(machine):
    machine(10)
    with budget.allot('campaign:T3', usd=1, provider='anthropic'):
        budget.allot('campaign:T3>item:1', usd=0.6)
        budget.allot('campaign:T3>item:2', usd=0.4)  # fits exactly: the control
        with pytest.raises(SpendCapReached) as refused:
            budget.allot('campaign:T3>item:3', usd=0.01)
    assert refused.value.scope == 'campaign:T3'


def test_a_task_that_does_not_fit_beside_the_standing_budgets_is_refused_at_open(machine):
    machine(10, standing='review = { provider = "anthropic", amount = 6, currency = "USD" }\n')
    budget.allot('campaign:A', usd=3, provider='anthropic')
    with pytest.raises(SpendCapReached) as refused:
        budget.allot('campaign:B', usd=2, provider='anthropic')
    assert refused.value.scope == 'anthropic'
    budget.close('campaign:A')
    budget.allot('campaign:B', usd=2, provider='anthropic')  # a closed task frees its carve-out


def test_standing_budgets_that_overrun_their_provider_cap_are_refused(machine):
    machine(1, standing='review = { provider = "anthropic", amount = 2, currency = "USD" }\n')
    with pytest.raises(ValueError, match='standing'):
        budget.standing()


def test_a_standing_name_is_never_opened_as_a_task(machine):
    machine(10, standing='review = { provider = "anthropic", amount = 2, currency = "USD" }\n')
    with pytest.raises(Unsupported, match='standing'):
        budget.allot('review', usd=1, provider='anthropic')


def test_a_split_derives_equal_shares_of_what_the_parent_leaves(machine):
    machine(10)
    task = budget.allot('campaign:T3', usd=1, provider='anthropic')
    shares = task.split(['arm:a', 'arm:b'])
    assert [s.name for s in shares] == ['campaign:T3>arm:a', 'campaign:T3>arm:b']
    assert [s.usd for s in shares] == [pytest.approx(0.5)] * 2


def test_every_ledger_row_carries_its_scope_and_a_scope_spent_is_its_subtree(machine):
    machine(10)
    with budget.allot('campaign:T3', usd=1, provider='anthropic'):
        _call()
        with budget.allot('campaign:T3>item:1', usd=0.1):
            _call()
    _call()
    rows = [json.loads(line) for line in budget.ledger_file().read_text('utf-8').splitlines()]
    assert [r['scope'] for r in rows] == ['campaign:T3', 'campaign:T3>item:1', None]
    assert budget.allot('campaign:T3').spent() == pytest.approx(2 * _CALL_USD)
    assert budget.allot('campaign:T3>item:1').spent() == pytest.approx(_CALL_USD)


def test_another_process_reopens_a_scope_by_name_and_reads_its_spent_from_the_ledger(machine, monkeypatch):
    machine(10)
    with budget.allot('campaign:T3', usd=1, provider='anthropic'):
        _call()
    probe = (
        'from ai_lab import budget\n'
        'budget.rate = lambda client, at=None: __import__("test_task_budgets")._RATE\n'
        'a = budget.allot("campaign:T3")\n'
        'print(a.usd, round(a.spent(), 6), a.provider)\n'
    )
    monkeypatch.syspath_prepend(str(__import__('pathlib').Path(__file__).parent))
    env = {**__import__('os').environ, 'PYTHONPATH': str(__import__('pathlib').Path(__file__).parent)}
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, env=env, check=True).stdout
    assert out.split() == ['1.0', str(_CALL_USD), 'anthropic']


def test_the_scope_rides_to_a_child_process_through_its_environment(machine):
    machine(10)
    with budget.allot('campaign:T3', usd=1, provider='anthropic'):
        out = subprocess.run(
            [sys.executable, '-c', f'import os; print(os.environ["{budget.SCOPE_ENV}"])'],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    assert out.strip() == 'campaign:T3'


def test_reopening_an_unknown_scope_is_refused_by_name(machine):
    machine(10)
    with pytest.raises(Unsupported, match='campaign:nope'):
        budget.allot('campaign:nope')


def test_a_scope_on_one_provider_refuses_a_call_on_another(machine):
    machine(10)
    with budget.allot('campaign:T3', usd=1, provider='ollama'), pytest.raises(Unsupported, match='ollama'):
        _call()


def test_the_cli_opens_a_child_of_a_standing_budget_and_refuses_one_past_it_with_exit_3(machine, capsys):
    machine(10, standing='review = { provider = "anthropic", amount = 2, currency = "USD" }\n')
    assert budget.main(['allot', 'review>o/r#5', '--usd', '1.5']) == 0
    assert json.loads(capsys.readouterr().out)['allotted_usd'] == 1.5
    assert budget.main(['allot', 'review>o/r#6', '--usd', '1']) == 3
    assert capsys.readouterr().err.startswith('review:')
    assert budget.main(['close', 'review>o/r#5']) == 0
    assert budget.main(['allot', 'review>o/r#6', '--usd', '1']) == 0
