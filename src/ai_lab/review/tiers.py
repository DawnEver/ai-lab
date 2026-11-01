"""The TIER LEDGER: one evidence row per review, so the tier bounds can be fitted from experiment later.

User ruling 2026-10-31: the bounds :func:`ai_lab.review.explore.tier` uses are INITIAL values, to be learned. Every
headless review appends one row (:func:`evidence`) to its repository's append-only JSONL, beside ai-lab's token
ledger (``state_dir()/review-tiers/<repo>.jsonl``); ``python -m ai_lab.review.tiers report <ledger>`` summarises
findings per thousand diff bytes (UTF-8) and cost, by tier and by size bucket. Nothing tunes itself: a person reads the
report and moves the constants.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from ai_lab.budget import state_dir
from ai_lab.review.explore import DIFF_LINE_BYTES, Changed, git

__all__ = ['BUCKETS', 'evidence', 'ledger_path', 'record', 'report']

#: Size buckets by diff UTF-8 BYTES (upper bounds, each 4x the last); past the last is its own bucket.
BUCKETS: Final = (4_000, 16_000, 64_000, 256_000, 1_024_000)

_SEVERITIES: Final = ('HIGH', 'MEDIUM', 'LOW')


def _read(repo: str, *args: str) -> str | None:
    """Git's stdout, or ``None`` when git cannot answer (a row is evidence, never a refused review)."""
    try:
        done = git(repo, *args)
    except ValueError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def ledger_path(repo: str) -> Path:
    """The ledger of *repo*: one JSONL per repository, named by its top-level directory."""
    top = _read(repo, 'rev-parse', '--show-toplevel') or str(Path(repo).resolve())
    return state_dir() / 'review-tiers' / f'{re.sub(r"[^\w.-]", "_", Path(top).name) or "repo"}.jsonl'


def evidence(
    *, range_: str, repo: str, rows: Sequence[Changed], tier: str, diff_bytes: int | None, turns: int,
    read_bytes: int, findings: Iterable[Mapping], seconds: float, usage: Mapping[str, int],
) -> dict:  # fmt: skip
    """One review's row: the change's size, the tier chosen, what reviewing it cost and found."""
    counts = dict.fromkeys(_SEVERITIES, 0)
    for finding in findings:
        if finding.get('severity') in counts:
            counts[finding['severity']] += 1
    commits = _read(repo, 'rev-list', '--count', '--end-of-options', range_)
    head = range_.rsplit('..', 1)[-1].lstrip('.') or 'HEAD'
    return {
        'at': datetime.now(UTC).isoformat(timespec='seconds'),
        'repo': Path(_read(repo, 'rev-parse', '--show-toplevel') or repo).name,
        'sha': _read(repo, 'rev-parse', '--verify', '--end-of-options', head),
        'range': range_,
        'files': len(rows),
        'lines': sum((a or 0) + (d or 0) for a, d, _ in rows),
        'diff_bytes': diff_bytes,
        'commits': int(commits) if commits and commits.isdigit() else None,
        'tier': tier,
        'turns': turns,
        'read_bytes': read_bytes,
        'findings': counts,
        'seconds': round(seconds, 3),
        'usage': dict(usage),
    }  # fmt: skip


def record(path: Path, row: Mapping) -> None:
    """Append *row* to the ledger at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps(row, ensure_ascii=False) + '\n')


def _bytes(row: Mapping) -> int:
    """The diff's UTF-8 bytes; a huge change's diff is never read, so its lines estimate them."""
    return row['diff_bytes'] if row.get('diff_bytes') is not None else row['lines'] * DIFF_LINE_BYTES


def _bucket(chars: int) -> str:
    for bound in BUCKETS:
        if chars <= bound:
            return f'<= {bound:,}'
    return f'> {BUCKETS[-1]:,}'


def _table(title: str, groups: Mapping[str, list[Mapping]]) -> list[str]:
    columns = ('group', 'reviews', *_SEVERITIES, 'findings/kB', 'tokens/review', 'read bytes/review',
               'turns/review', 's/review')  # fmt: skip
    lines = [f'## {title}', '', '| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    for name, rows in groups.items():
        n = len(rows)
        found = {s: sum(r['findings'].get(s, 0) for r in rows) for s in _SEVERITIES}
        kbytes = sum(_bytes(r) for r in rows) / 1000
        tokens = sum(sum(r['usage'].get(k, 0) for k in ('input_tokens', 'output_tokens')) for r in rows)
        per_kb = sum(found.values()) / kbytes if kbytes else 0.0
        cells = [name, str(n), *(str(found[s]) for s in _SEVERITIES), f'{per_kb:.3f}', f'{tokens / n:.0f}',
                 f'{sum(r["read_bytes"] for r in rows) / n:.0f}', f'{sum(r["turns"] for r in rows) / n:.1f}',
                 f'{sum(r["seconds"] for r in rows) / n:.1f}']  # fmt: skip
        lines.append('| ' + ' | '.join(cells) + ' |')
    return [*lines, '']


def report(rows: Sequence[Mapping]) -> str:
    """The Markdown summary of ledger *rows*: by tier, then by diff-size bucket."""
    by_tier: dict[str, list[Mapping]] = defaultdict(list)
    by_size: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_tier[row['tier']].append(row)
    for row in sorted(rows, key=_bytes):
        by_size[_bucket(_bytes(row))].append(row)
    head = [f'# Review tier ledger: {len(rows)} review(s)', '']
    return '\n'.join(head + _table('By tier', dict(sorted(by_tier.items()))) + _table('By diff size', by_size))


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m ai_lab.review.tiers report <ledger>``: print the ledger's summary."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.review.tiers', description=__doc__.split('\n', 1)[0])
    verbs = parser.add_subparsers(dest='verb', required=True)
    verbs.add_parser('report', help='summarise a ledger by tier and size').add_argument('ledger')
    args = parser.parse_args(argv)
    text = Path(args.ledger).read_text(encoding='utf-8')
    sys.stdout.write(report([json.loads(line) for line in text.splitlines() if line.strip()]) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
