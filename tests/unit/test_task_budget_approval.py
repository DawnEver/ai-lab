"""A declared task budget is a DEFAULT; one task may be APPROVED for a larger allotment (user ruling 2026-11-06).

An approval is a row (who, when, why, amount) in the same state store as allotments; ``allot`` uses it for that
exact task name, else the caller's default; an approval past what the provider month leaves beside the standing
and open allotments is refused at approve, naming the level; ``revoke`` restores the default.
"""

import json

import pytest

from ai_lab import SpendCapReached, budget, spend
from ai_lab.prices import Price, Quote

_RATE = Quote(Price(1.0, 2.0), 'test', agreed=True, sources={})


@pytest.fixture
def machine(tmp_path, monkeypatch, keys):
    path = tmp_path / 'machine.toml'
    path.write_text(
        '[ai_lab.spend_cap]\nanthropic = { amount = 10, currency = "USD" }\n'
        '[ai_lab.standing]\nreview = { provider = "anthropic", amount = 4, currency = "USD" }\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('LAB_COMMONS_CONFIG', str(path))
    monkeypatch.setattr(budget, 'rate', lambda client, at=None: _RATE)
    monkeypatch.delenv(budget.SCOPE_ENV, raising=False)
    return path


def _approve(task='campaign:T3:anthropic', amount=5.0):
    return budget.approve(task, amount, provider='anthropic', by='mb', reason='big sweep')


def test_without_an_approval_allot_opens_the_callers_default(machine):
    opened = budget.allot('campaign:T3:anthropic', usd=1, provider='anthropic')
    assert (opened.usd, opened.default_usd) == (1, None)


def test_an_approved_task_opens_at_its_approved_amount_and_records_who_and_why(machine):
    _approve()
    opened = budget.allot('campaign:T3:anthropic', usd=1, provider='anthropic')
    assert (opened.usd, opened.default_usd) == (5, 1)
    row = budget.approvals()['campaign:T3:anthropic']
    assert (row.usd, row.by, row.reason, row.provider) == (5, 'mb', 'big sweep', 'anthropic')
    assert row.at


def test_planted_control_an_approval_for_another_task_does_not_apply(machine):
    _approve('campaign:T4:anthropic')
    assert budget.allot('campaign:T3:anthropic', usd=1, provider='anthropic').usd == 1


def test_an_approval_past_the_room_beside_the_standing_budgets_is_refused_naming_the_provider(machine):
    with pytest.raises(SpendCapReached) as refused:
        _approve(amount=6.5)  # 10 cap - 4 review = 6 free
    assert refused.value.scope == 'anthropic'
    assert budget.approvals() == {}


def test_revoke_restores_the_default(machine):
    _approve()
    budget.revoke('campaign:T3:anthropic')
    assert budget.approvals() == {}
    assert budget.allot('campaign:T3:anthropic', usd=1, provider='anthropic').usd == 1


def test_an_approval_in_another_currency_converts_through_the_rate(machine):
    machine.write_text(machine.read_text('utf-8') + '[ai_lab.usd_per]\nCNY = 0.5\n', encoding='utf-8')
    budget.approve('t', 8, 'CNY', provider='anthropic', by='mb', reason='r')
    assert budget.approvals()['t'].usd == 4


def test_the_spend_report_shows_approved_against_default(machine):
    _approve()
    budget.allot('campaign:T3:anthropic', usd=1, provider='anthropic')
    row = spend.budgets()['campaign:T3:anthropic']
    assert (row['allotted_usd'], row['default_usd']) == (5, 1)


def test_the_cli_approves_lists_and_revokes(machine, capsys):
    argv = ['approve', 't', '--amount', '2', '--provider', 'anthropic', '--by', 'mb', '--reason', 'r']
    assert budget.main(argv) == 0
    assert budget.main(['approvals']) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['t']['usd'] == 2
    assert budget.main(['approve', 't', '--amount', '99', '--provider', 'anthropic', '--reason', 'r']) == 3
    assert budget.main(['revoke', 't']) == 0
    assert budget.approvals() == {}
