"""The family's model spend in US dollars: the token ledger priced at each call's time (user ruling 2026-10-31).

ONE BUDGET, DECLARED ONCE, in the machine file's ``[ai_lab.spend]`` table (:mod:`lab_commons.config`), here
because spend is MEASURED here, where calls are made (user ruling 2026-11-03)::

    [ai_lab.spend]
    monthly_usd = 1000   # every model call of the family in one month
    reserve_usd = 250    # held back: drawn only by an explicit raise; the rest (750) is allocated
    reset_day = 1        # the day of the month (1-28, UTC) a month opens on

The report states the three; what one PR may spend of the allocated share, and how a raise draws on the
reserve, are a policy of its runner (dev-lab's review budget), which reads this report and restates no number.

ONE LEDGER. The source is :mod:`ai_lab.budget`'s token ledger (one row per call: tokens, model, time, tag); dollars
are derived from it by :func:`ai_lab.prices.quote_for` at the call's time and never stored, so a corrected price
re-prices history. A row this reader cannot price (a legacy row with no model, a model no catalog knows) is
counted in ``unpriced_calls``, never as free.

The ledger is machine-local (:func:`ai_lab.budget.state_dir`), so a family served by several hosts reads each
host's share only: a report is THIS host's month, and an ephemeral runner's is its own run.

``python -m ai_lab.spend --json`` prints :func:`report`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

from lab_commons.config import section

from ai_lab import budget
from ai_lab.prices import Quote, cost, quote_for
from ai_lab.usage import Usage

__all__ = [
    'DEFAULT_MONTHLY_USD',
    'DEFAULT_RESERVE_USD',
    'DEFAULT_RESET_DAY',
    'LOOKBACK_DAYS',
    'SPEND_SCHEMA',
    'month_start',
    'report',
]

#: The family's model budget per month, every call of every repo: user ruling 2026-10-31.
DEFAULT_MONTHLY_USD: Final = 1000.0

#: The share of it held back as a reserve, drawn only on demand: user ruling 2026-11-03 (750 allocated, 250 reserve).
DEFAULT_RESERVE_USD: Final = 250.0

#: The day of the month a month opens on (UTC): user ruling 2026-11-03, the 1st. At most 28, so every month has it.
DEFAULT_RESET_DAY: Final = 1

TABLE: Final = 'spend'
_KEYS: Final = frozenset({'monthly_usd', 'reserve_usd', 'reset_day'})

#: How far back the per-tag history a cap is FITTED from reaches: three months, enough PRs to read a quantile
#: from and recent enough that the family's habits still hold.
LOOKBACK_DAYS: Final = 90

#: The report's version: a reader refuses one it does not know. Printed, never stored. 2: ``reserve_usd``, and
#: ``month`` is the opening date (``YYYY-MM-DD``) of a month that opens on its reset day.
SPEND_SCHEMA: Final = 2

Price = Callable[[str, datetime | None], Quote | None]


def _default_price(client: str, at: datetime | None) -> Quote | None:
    """The rate at *at*; the current one (catalogs read once) when no snapshot that old holds the model."""
    return quote_for(client, at=at) or quote_for(client)


def _amount(table: dict, key: str, default: float) -> float:
    value = table.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
        msg = f'[{budget.APP}.{TABLE}] {key} must be a non-negative number of US dollars, got {value!r}'
        raise ValueError(msg)
    return float(value)


def _declared() -> tuple[float, float, int]:
    """The machine file's ``(monthly_usd, reserve_usd, reset_day)``, each defaulted; a malformed one refused by name."""
    table = section(budget.APP).get(TABLE, {})
    if not isinstance(table, dict):
        msg = f'[{budget.APP}.{TABLE}] is a table, got {table!r}'
        raise ValueError(msg)  # noqa: TRY004 -- a malformed machine file is a value error, not a type error
    if unknown := sorted(set(table) - _KEYS):
        msg = f'[{budget.APP}.{TABLE}] has unknown key(s) {unknown}; the keys are {sorted(_KEYS)}'
        raise ValueError(msg)
    monthly = _amount(table, 'monthly_usd', DEFAULT_MONTHLY_USD)
    reserve = _amount(table, 'reserve_usd', DEFAULT_RESERVE_USD)
    if reserve > monthly:
        msg = f'[{budget.APP}.{TABLE}] reserve_usd {reserve} exceeds monthly_usd {monthly}'
        raise ValueError(msg)
    day = table.get('reset_day', DEFAULT_RESET_DAY)
    if not isinstance(day, int) or isinstance(day, bool) or not 1 <= day <= 28:  # noqa: PLR2004 -- every month has a 28th
        msg = f'[{budget.APP}.{TABLE}] reset_day must be a whole day of the month 1-28, got {day!r}'
        raise ValueError(msg)
    return monthly, reserve, day


def month_start(today: date, reset_day: int) -> date:
    """The day the month holding *today* opened: its *reset_day*, else the previous month's."""
    if today.day >= reset_day:
        return today.replace(day=reset_day)
    previous = today.replace(day=1) - timedelta(days=1)
    return previous.replace(day=reset_day)


def _rows(directory: Path, since: date) -> list[tuple[date, dict]]:
    found = []
    for path in sorted(directory.glob('tokens-*.jsonl')):
        try:
            day = date.fromisoformat(path.stem.removeprefix('tokens-'))
        except ValueError:
            continue
        if day >= since:
            found += [(day, json.loads(line)) for line in path.read_text('utf-8').splitlines() if line.strip()]
    return found


def _usd(row: dict, price: Price) -> float | None:
    if not row.get('model'):
        return None
    at = datetime.fromisoformat(row['at']) if row.get('at') else None
    usage = Usage(row.get('input', 0), row.get('output', 0), row.get('cached', 0))
    quote = price(f'{row["provider"]}:{row["model"]}', at)
    return cost(usage, None if quote is None else quote.price)


def report(*, today: date | None = None, directory: Path | None = None, price: Price | None = None) -> dict:
    """The month's spend against the declared budget and reserve, per tag, and the per-tag history to fit caps from."""
    monthly, reserve, reset_day = _declared()
    today = datetime.now(UTC).date() if today is None else today
    directory = budget.state_dir() if directory is None else directory
    price = _default_price if price is None else price
    opened = month_start(today, reset_day)
    rows = _rows(directory, min(opened, today - timedelta(days=LOOKBACK_DAYS - 1)))
    window = [day for day, _ in rows if day > today - timedelta(days=LOOKBACK_DAYS)]
    spent, unpriced = 0.0, 0
    month_by_tag: dict[str, float] = defaultdict(float)
    window_by_tag: dict[str, float] = defaultdict(float)
    for day, row in rows:
        usd = _usd(row, price)
        in_month = day >= opened
        if usd is None:
            unpriced += in_month
            continue
        tag = row.get('tag')
        if in_month:
            spent += usd
            if tag:
                month_by_tag[tag] += usd
        if tag and day > today - timedelta(days=LOOKBACK_DAYS):
            window_by_tag[tag] += usd
    first = min(window, default=today)
    return {
        'schema': SPEND_SCHEMA, 'budget_usd': monthly, 'reserve_usd': reserve, 'month': opened.isoformat(),
        'spent_usd': spent, 'unpriced_calls': unpriced, 'month_by_tag': dict(month_by_tag),
        'window': {'days': (today - first).days + 1, 'by_tag': dict(window_by_tag)},
    }  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    """Print :func:`report`: JSON with ``--json``, else one line per figure."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.spend', description=__doc__.splitlines()[0])
    parser.add_argument('--json', action='store_true', help='print the report as JSON')
    args = parser.parse_args(argv)
    got = report(price=_default_price)
    if args.json:
        sys.stdout.write(json.dumps(got, indent=1) + '\n')
        return 0
    head = (
        f'month from {got["month"]}: {got["spent_usd"]:.2f} of {got["budget_usd"]:.2f} USD, '
        f'{got["reserve_usd"]:.2f} of it the reserve ({got["unpriced_calls"]} unpriced)'
    )
    tags = sorted(got['month_by_tag'].items(), key=lambda kv: -kv[1])
    sys.stdout.write('\n'.join([head, *(f'  {usd:9.2f}  {tag}' for tag, usd in tags)]) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
