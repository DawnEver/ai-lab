"""The family's model spend in US dollars: the token ledger priced at each call's time (user ruling 2026-10-31).

ONE BUDGET, DECLARED ONCE. :data:`FAMILY_MONTHLY_USD` is every model call of the family in one calendar month (UTC),
declared here because spend is MEASURED here, where calls are made; what one PR may spend of it is a policy of
its runner (dev-lab's review budget), which reads this report and never restates the number.

ONE LEDGER. The source is :mod:`ai_lab.budget`'s token ledger (one row per call: tokens, model, time, tag); dollars
are derived from it by :func:`ai_lab.prices.quote_for` at the call's time and never stored, so a corrected price
re-prices history. A row this reader cannot price (a legacy row with no model, a model no catalog knows) is
counted in ``unpriced_calls``, never as free.

The ledger is machine-local (:func:`ai_lab.budget.state_dir`), so a family served by several hosts reads each
host's share; the review runner sums what its hosts report.

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

from ai_lab import budget
from ai_lab.prices import Quote, cost, quote_for
from ai_lab.usage import Usage

__all__ = ['FAMILY_MONTHLY_USD', 'LOOKBACK_DAYS', 'SPEND_SCHEMA', 'report']

#: The family's model budget per calendar month (UTC), every call of every repo: user ruling 2026-10-31.
FAMILY_MONTHLY_USD: Final = 1000.0

#: How far back the per-tag history a cap is FITTED from reaches: three months, enough PRs to read a quantile
#: from and recent enough that the family's habits still hold.
LOOKBACK_DAYS: Final = 90

#: The report's version: a reader refuses one it does not know. Printed, never stored.
SPEND_SCHEMA: Final = 1

Price = Callable[[str, datetime | None], Quote | None]


def _default_price(client: str, at: datetime | None) -> Quote | None:
    """The rate at *at*; the current one (catalogs read once) when no snapshot that old holds the model."""
    return quote_for(client, at=at) or quote_for(client)


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
    """The month's spend against :data:`FAMILY_MONTHLY_USD`, per tag, and the per-tag history to fit caps from."""
    today = datetime.now(UTC).date() if today is None else today
    directory = budget.state_dir() if directory is None else directory
    price = _default_price if price is None else price
    month_start = today.replace(day=1)
    rows = _rows(directory, min(month_start, today - timedelta(days=LOOKBACK_DAYS - 1)))
    window = [day for day, _ in rows if day > today - timedelta(days=LOOKBACK_DAYS)]
    spent, unpriced = 0.0, 0
    month_by_tag: dict[str, float] = defaultdict(float)
    window_by_tag: dict[str, float] = defaultdict(float)
    for day, row in rows:
        usd = _usd(row, price)
        in_month = day >= month_start
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
        'schema': SPEND_SCHEMA, 'budget_usd': FAMILY_MONTHLY_USD, 'month': month_start.strftime('%Y-%m'),
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
    head = f'{got["month"]}: {got["spent_usd"]:.2f} of {got["budget_usd"]:.2f} USD ({got["unpriced_calls"]} unpriced)'
    tags = sorted(got['month_by_tag'].items(), key=lambda kv: -kv[1])
    sys.stdout.write('\n'.join([head, *(f'  {usd:9.2f}  {tag}' for tag, usd in tags)]) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
