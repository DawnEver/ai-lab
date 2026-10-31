"""EXPLORATORY REVIEW: the reviewer is given the change's MAP and reads what it chooses, never the whole diff inlined.

User ruling 2026-10-31: a review must happen however large the PR, and it must not, by default, dump the whole
diff into the prompt -- the reviewer explores. The headless run (:mod:`ai_lab.review.run`) therefore sends:

* the change map (:func:`change_map`): the range, every changed file with its added/deleted line counts, and the
  directory tree of what changed. Past :data:`MAX_LISTED_FILES` files the map is the DIRECTORY level only, and the
  reviewer lists a directory to see its files;
* each turn, the reviewer answers with ``findings`` (done) or ``requests`` (:data:`REQUESTS`), which
  :func:`serve` answers from git -- the change to a path, a file at the head, a ``git grep`` over the head, the
  changed files under a directory. ``show`` and ``grep`` reach the WHOLE head, not only the changed files, so a
  call can be followed into code the change did not touch. A turn's answers total at most :data:`TURN_BYTES`; one
  past what remains is cut to it and the cut is said, never dropped.

This runs on the one ``respond`` verb every provider wire already speaks: no tool-use API, no second client.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Final

__all__ = ['MAX_LISTED_FILES', 'REQUESTS', 'TURNS', 'TURN_BYTES', 'Changed', 'change_map', 'numstat', 'serve']

#: How many turns a reviewer has: every one but the last may ask; the last must answer with findings.
TURNS: Final = 6

#: Past this many changed files the map lists directories only; ``list`` opens one.
MAX_LISTED_FILES: Final = 150

#: What one turn's answers may total; past it the rest is cut and the cut is said.
TURN_BYTES: Final = 120_000

#: What a reviewer may ask for, each a list of strings, and what each returns.
REQUESTS: Final = {
    'diff': 'the change to these paths in the range',
    'show': 'these files as they are at the head',
    'grep': 'git grep -n of these patterns over the head',
    'list': 'the changed files under these directories, with their line counts',
}

#: One changed file: lines added, lines deleted (``None`` for a binary file), its path at the head.
Changed = tuple[int | None, int | None, str]


def _git(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    git = shutil.which('git')
    if git is None:
        msg = 'git is not on PATH: a review reads its change with git'
        raise ValueError(msg)
    return subprocess.run(
        [git, '-C', repo, *args], capture_output=True, text=True, encoding='utf-8', errors='replace', check=False
    )


def numstat(range_: str, repo: str, paths: Sequence[str] = ()) -> list[Changed]:
    """The files *range_* changes (restricted to *paths* when any), renames followed; ``ValueError`` naming git's."""
    done = _git(repo, 'diff', '--numstat', '-M', '--end-of-options', range_, *(['--', *paths] if paths else []))
    if done.returncode:
        msg = f'git diff {range_} in {repo} failed: {done.stderr.strip()}'
        raise ValueError(msg)
    rows = []
    for line in filter(None, done.stdout.splitlines()):
        added, deleted, path = line.split('\t', 2)
        if ' => ' in path:  # a rename: {a => b}/c or a => b; the head's path is what the reviewer reads
            head, _, tail = path.partition('{')
            if tail:
                inner, _, rest = tail.partition('}')
                path = head + inner.split(' => ')[1] + rest
            else:
                path = path.split(' => ')[1]
            path = path.replace('//', '/')
        rows.append((None if added == '-' else int(added), None if deleted == '-' else int(deleted), path))
    return rows


def _counts(added: int | None, deleted: int | None) -> str:
    return 'binary' if added is None or deleted is None else f'+{added} -{deleted}'


def _directories(rows: Sequence[Changed]) -> dict[str, list[Changed]]:
    found: dict[str, list[Changed]] = defaultdict(list)
    for row in rows:
        found[str(PurePosixPath(row[2]).parent)].append(row)
    return dict(sorted(found.items()))


def change_map(range_: str, rows: Sequence[Changed], *, max_files: int = MAX_LISTED_FILES) -> str:
    """The change map a reviewer starts from: the totals, the directory tree, and the files unless there are many."""
    added = sum(row[0] or 0 for row in rows)
    deleted = sum(row[1] or 0 for row in rows)
    tree = _directories(rows)
    lines = [f'{range_}: {len(rows)} file(s) changed, +{added} -{deleted}, in {len(tree)} director(ies).', '']
    lines.append('Directories (files changed, lines):')
    for directory, inside in tree.items():
        plus = sum(row[0] or 0 for row in inside)
        minus = sum(row[1] or 0 for row in inside)
        lines.append(f'  {directory}/  {len(inside)} file(s)  +{plus} -{minus}')
    if len(rows) > max_files:
        lines += ['', f'{len(rows)} files is past {max_files}: ask `list` for a directory to see its files.']
    else:
        lines += ['', 'Files:'] + [f'  {_counts(a, d):>14}  {path}' for a, d, path in rows]
    return '\n'.join(lines)


def _head(range_: str) -> str:
    """The revision a range ends at: ``b`` of ``a..b`` / ``a...b``, else the range itself."""
    return range_.rsplit('..', 1)[-1].lstrip('.') or 'HEAD'


def _answer(kind: str, item: str, range_: str, repo: str, rows: Sequence[Changed]) -> str:
    if kind == 'list':
        prefix = item.strip('/')
        inside = [row for row in rows if prefix in {'', '.'} or row[2].startswith(prefix + '/')]
        return '\n'.join(f'{_counts(a, d):>14}  {path}' for a, d, path in inside) or '(no changed file under it)'
    if kind == 'diff':
        done = _git(repo, 'diff', '-M', '--end-of-options', range_, '--', item)
    elif kind == 'show':
        done = _git(repo, 'show', f'{_head(range_)}:{item}')
    else:
        done = _git(repo, 'grep', '-n', '-I', '-e', item, _head(range_))
    if done.returncode and not done.stdout:
        return f'(nothing: {done.stderr.strip() or "no match"})'
    return done.stdout or '(empty)'


def _listed(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def serve(
    requests: Mapping[str, object], *, range_: str, repo: str, rows: Sequence[Changed], budget: int = TURN_BYTES
) -> str:
    """The answers to one turn's *requests*, in :data:`REQUESTS` order, together at most *budget* characters.

    An answer past what remains is CUT to it, never dropped (a file larger than a turn is still read in part), and
    the cut names how much was shown and every request left unanswered, so the reviewer can narrow and ask again.
    """
    asked = [(kind, str(item)) for kind in REQUESTS for item in _listed(requests.get(kind))]
    parts, used = [], 0
    for at, (kind, item) in enumerate(asked):
        text = f'### {kind} {item}\n{_answer(kind, item, range_, repo, rows)}\n'
        if used + len(text) > budget:
            shown = max(budget - used, 0)
            left = ', '.join(f'{k} {i}' for k, i in asked[at + 1 :]) or 'none'
            parts.append(text[:shown])
            parts.append(f'\n### cut: {kind} {item} shown to {shown} of {len(text)} characters (this turn holds '
                         f'{budget}); not answered: {left}. Ask a narrower grep, or the rest, next turn.\n')  # fmt: skip
            return ''.join(parts)
        parts.append(text)
        used += len(text)
    unknown = sorted(set(requests) - set(REQUESTS))
    if unknown:
        parts.append(f'### refused: {unknown} are not requests; ask only {sorted(REQUESTS)}\n')
    return ''.join(parts) or '(no request named anything to read)'
