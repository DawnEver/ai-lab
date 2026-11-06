"""The family's model spend in US dollars: the token ledger priced at each call's time (user ruling 2026-10-31).

EVERY BUDGET IS DECLARED ONCE, in :mod:`ai_lab.budget`: a provider's month (``[ai_lab.spend_cap]``), a standing
allotment carved out of it (``[ai_lab.standing]``: ``review``, a CI job) and each open task allotment. The report's
``budgets`` restates none of them: it reads each one's allotted and spent from there. What one PR may spend of
``review`` is its runner's policy (dev-lab's review budget), opened as a child allotment ``review>owner/repo#n``.

ONE LEDGER. The source is :mod:`ai_lab.budget`'s token ledger (one row per call: tokens, model, time, tag); dollars
are derived from it by :func:`ai_lab.budget.row_usd` at the call's time -- the same pricing a provider's
``[ai_lab.spend_cap]`` is enforced with -- and never stored, so a corrected price
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

from ai_lab import budget
from ai_lab.prices import Quote

__all__ = ['LOOKBACK_DAYS', 'SPEND_SCHEMA', 'budgets', 'report']

#: How far back the per-tag history a cap is FITTED from reaches: three months, enough PRs to read a quantile
#: from and recent enough that the family's habits still hold.
LOOKBACK_DAYS: Final = 90

#: The report's version: a reader refuses one it does not know. Printed, never stored. 2: ``budgets`` per level
#: replaced the one family budget.
SPEND_SCHEMA: Final = 2

Price = Callable[[str, datetime | None], Quote | None]


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


def budgets() -> dict[str, dict]:
    """Every standing and open task allotment by path, with what it was allotted and has spent.

    Per path: ``provider``, ``allotted_usd``, ``spent_usd``, ``standing``, and ``default_usd`` -- the declared
    default an approval replaced (``None``: the allotment is its default).
    """
    held = {**budget.task_allotments(), **budget.standing()}
    return {
        name: {
            'provider': a.provider,
            'allotted_usd': a.usd,
            'spent_usd': a.spent(),
            'standing': a.standing,
            'default_usd': a.default_usd,
        }
        for name, a in sorted(held.items())
    }


def report(*, today: date | None = None, directory: Path | None = None, price: Price | None = None) -> dict:
    """The month's spend, per tag, the per-tag history to fit caps from, and every allotment (:func:`budgets`)."""
    today = datetime.now(UTC).date() if today is None else today
    directory = budget.state_dir() if directory is None else directory
    month_start = today.replace(day=1)
    rows = _rows(directory, min(month_start, today - timedelta(days=LOOKBACK_DAYS - 1)))
    window = [day for day, _ in rows if day > today - timedelta(days=LOOKBACK_DAYS)]
    spent, unpriced = 0.0, 0
    month_by_tag: dict[str, float] = defaultdict(float)
    window_by_tag: dict[str, float] = defaultdict(float)
    for day, row in rows:
        usd = budget.row_usd(row, price)
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
        'schema': SPEND_SCHEMA, 'budgets': budgets(), 'month': month_start.strftime('%Y-%m'),
        'spent_usd': spent, 'unpriced_calls': unpriced, 'month_by_tag': dict(month_by_tag),
        'window': {'days': (today - first).days + 1, 'by_tag': dict(window_by_tag)},
    }  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    """Print :func:`report`: JSON with ``--json``, else one line per figure."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.spend', description=__doc__.splitlines()[0])
    parser.add_argument('--json', action='store_true', help='print the report as JSON')
    args = parser.parse_args(argv)
    got = report()
    if args.json:
        sys.stdout.write(json.dumps(got, indent=1) + '\n')
        return 0
    head = f'{got["month"]}: {got["spent_usd"]:.2f} USD ({got["unpriced_calls"]} unpriced)'
    tags = sorted(got['month_by_tag'].items(), key=lambda kv: -kv[1])
    held = [
        f'  {b["spent_usd"]:9.2f} / {b["allotted_usd"]:9.2f}  {name}'
        + ('' if b['default_usd'] is None else f'  (approved; default {b["default_usd"]:.2f})')
        for name, b in got['budgets'].items()
    ]
    sys.stdout.write('\n'.join([head, *(f'  {usd:9.2f}  {tag}' for tag, usd in tags), *held]) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
