"""``python -m ai_lab.report LEDGER [LEDGER ...]`` -- calls, latency, tokens and cost per client.

Read-only over ledger files: one row per client name (``provider:model[@effort]``), so two efforts
of one model, or two models on one task, are two rows to compare. Each entry is priced at the rate
in force at its own time (:mod:`ai_lab.prices`); the ``price`` column says where the rate came from
and flags a catalog DISPUTE. Tokens are blank where the wire reports none (an entry recorded before
usage was read counts as unreported).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ai_lab.prices import Quote, cost, quote_for
from ai_lab.usage import Usage

__all__ = ['ClientSummary', 'summarize']


@dataclass
class ClientSummary:
    """Totals for one client over every entry read."""

    client: str
    calls: int = 0
    questions: int = 0
    latency_s: float = 0.0
    reported: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = 0.0
    price: str = '-'

    @property
    def mean_latency_s(self) -> float:
        """Mean seconds per call."""
        return self.latency_s / self.calls if self.calls else 0.0


def summarize(paths: Iterable[Path], *, price: Callable[..., Quote | None] = quote_for) -> list[ClientSummary]:
    """One :class:`ClientSummary` per client name across ``paths``, in first-seen order."""
    rows: dict[str, ClientSummary] = {}
    quotes: dict[tuple[str, str], Quote | None] = {}
    for path in paths:
        for line in Path(path).read_text('utf-8').splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            name = entry.get('client') or f'{entry["provider"]}:{entry["model"]}'
            row = rows.setdefault(name, ClientSummary(name))
            row.calls += 1
            row.questions += len(entry.get('answers', ()))
            row.latency_s += entry.get('latency_s', 0.0)
            if entry.get('usage') is None:
                row.cost_usd = None
                continue
            usage = Usage(**entry['usage'])
            row.reported += 1
            row.input_tokens += usage.input_tokens
            row.cached_tokens += usage.cached_tokens
            row.output_tokens += usage.output_tokens
            row.reasoning_tokens += usage.reasoning_tokens
            at = datetime.fromisoformat(entry['at']) if entry.get('at') else None
            memo = (name, '' if at is None else at.date().isoformat())
            if memo not in quotes:
                quotes[memo] = price(name, at=at)
            held = quotes[memo]
            if held is not None:
                row.price = f'{held.origin}{"" if held.agreed else " DISPUTED"}'
            dollars = cost(usage, None if held is None else held.price)
            row.cost_usd = None if dollars is None or row.cost_usd is None else row.cost_usd + dollars
    return list(rows.values())


def _table(rows: Sequence[ClientSummary]) -> str:
    head = ('client', 'calls', 'questions', 'mean s', 'usage', 'input', 'cached', 'output', 'reasoning', 'USD', 'price')
    body = [
        (
            r.client,
            str(r.calls),
            str(r.questions),
            f'{r.mean_latency_s:.2f}',
            f'{r.reported}/{r.calls}',
            str(r.input_tokens),
            str(r.cached_tokens),
            str(r.output_tokens),
            str(r.reasoning_tokens),
            '-' if r.cost_usd is None else f'{r.cost_usd:.4f}',
            r.price,
        )
        for r in rows
    ]
    widths = [max(len(c[i]) for c in (head, *body)) for i in range(len(head))]
    return '\n'.join('  '.join(cell.rjust(w) for cell, w in zip(line, widths, strict=True)) for line in (head, *body))


def main(argv: Sequence[str] | None = None) -> int:
    """Print the table for the ledgers named on the command line."""
    paths = [Path(p) for p in (sys.argv[1:] if argv is None else argv)]
    if not paths:
        sys.stderr.write('usage: python -m ai_lab.report LEDGER [LEDGER ...]\n')
        return 2
    sys.stdout.write(_table(summarize(paths)) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
