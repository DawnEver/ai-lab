"""The month's spend: every call's tokens priced at its time, summed per month and per tag (user ruling 2026-10-31)."""

import json
from datetime import UTC, date, datetime

import pytest

from ai_lab import ResponseRequest, Text, budget, connect, spend
from ai_lab.prices import Price, Quote

_REQUEST = ResponseRequest('i', (Text('t'),), {'type': 'object', 'properties': {}})
_ONE_PER_MILLION = Quote(Price(1.0, 2.0, 0.5), 'test', agreed=True, sources={})


def _price(client, at):
    return _ONE_PER_MILLION if client == 'anthropic:claude-x' else None


def _reply(inp, out, cached=0):
    def post(call, **_):
        usage = {'input_tokens': inp, 'output_tokens': out, 'cache_read_input_tokens': cached}
        return {'content': [{'type': 'text', 'text': '{}'}], 'usage': usage}

    return post


def _write(directory, day, *rows):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'tokens-{day}.jsonl').open('a', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row) + '\n')


def _row(tag, inp, out=0, model='claude-x', at='2026-10-05T00:00:00+00:00'):
    return {'provider': 'anthropic', 'key': None, 'tokens': inp + out, 'model': model, 'input': inp, 'output': out,
            'cached': 0, 'at': at, 'tag': tag}  # fmt: skip


def test_a_call_records_its_model_tokens_time_and_the_tag_it_is_charged_to(keys, monkeypatch):
    monkeypatch.setenv(budget.TAG_ENV, 'DawnEver/dev-lab#11')
    connect('anthropic:claude-x', post=_reply(700, 300, cached=200)).respond(_REQUEST)
    (row,) = [json.loads(line) for line in budget.ledger_file().read_text('utf-8').splitlines()]
    assert row['tag'] == 'DawnEver/dev-lab#11'
    assert (row['model'], row['input'], row['output'], row['cached'], row['tokens']) == (
        'claude-x',
        900,
        300,
        200,
        1200,
    )
    assert datetime.fromisoformat(row['at']).tzinfo is not None


def test_planted_control_an_untagged_call_is_charged_to_no_tag(keys, monkeypatch):
    monkeypatch.delenv(budget.TAG_ENV, raising=False)
    connect('anthropic:claude-x', post=_reply(1, 1)).respond(_REQUEST)
    (row,) = [json.loads(line) for line in budget.ledger_file().read_text('utf-8').splitlines()]
    assert row['tag'] is None


def test_the_report_prices_each_call_and_sums_the_month_and_each_tag(tmp_path):
    _write(tmp_path, '2026-09-30', _row('r#1', 1_000_000))  # last month: in the window, not the month
    _write(tmp_path, '2026-10-05', _row('r#1', 1_000_000, 1_000_000), _row('r#2', 2_000_000))
    _write(tmp_path, '2026-10-06', _row(None, 1_000_000), _row('r#2', 1, model='unpriced-model'))
    got = spend.report(today=date(2026, 10, 6), directory=tmp_path, price=_price)
    assert got['budget_usd'] == spend.FAMILY_MONTHLY_USD == 1000.0
    assert got['month'] == '2026-10'
    assert got['spent_usd'] == pytest.approx(3.0 + 2.0 + 1.0)
    assert got['month_by_tag'] == pytest.approx({'r#1': 3.0, 'r#2': 2.0})
    assert got['window']['by_tag'] == pytest.approx({'r#1': 4.0, 'r#2': 2.0})
    assert got['window']['days'] == 7, 'from the first ledger day in the lookback to today'
    assert got['unpriced_calls'] == 1


def test_a_legacy_row_without_a_model_is_unpriced_never_free_and_never_a_crash(tmp_path):
    _write(tmp_path, '2026-10-06', {'provider': 'anthropic', 'key': None, 'tokens': 10})
    got = spend.report(today=date(2026, 10, 6), directory=tmp_path, price=_price)
    assert (got['spent_usd'], got['unpriced_calls']) == (0.0, 1)


def test_a_row_is_priced_at_its_own_time(tmp_path):
    seen = []

    def price(client, at):
        seen.append(at)
        return _ONE_PER_MILLION

    _write(tmp_path, '2026-10-05', _row('r#1', 1, at='2026-10-05T12:00:00+00:00'))
    spend.report(today=date(2026, 10, 6), directory=tmp_path, price=price)
    assert seen == [datetime(2026, 10, 5, 12, tzinfo=UTC)]


def test_the_cli_prints_the_report_as_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(spend, '_default_price', _price)
    monkeypatch.setattr(budget, 'state_dir', lambda: tmp_path)
    _write(tmp_path, datetime.now(UTC).date().isoformat(), _row('r#1', 1_000_000, at=datetime.now(UTC).isoformat()))
    assert spend.main(['--json']) == 0
    got = json.loads(capsys.readouterr().out)
    assert got['schema'] == spend.SPEND_SCHEMA
    assert got['month_by_tag'] == {'r#1': 1.0}
