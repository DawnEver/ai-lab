"""SIZE-TIERED REVIEW: a small change is read whole, a larger one explored from its map, a huge one by directory.

User rulings 2026-10-31: every PR is reviewed however large; "a small diff is reviewed with the full diff
directly; a large one opens agent mode; beyond that only the directory and the commit messages". :func:`tier` is
the ONE place the tier is decided, and :func:`brief` the one place its prompt payload is built:

* ``small`` -- the diff is at most :data:`INLINE_BYTES` UTF-8 bytes: the prompt carries it whole;
* ``medium`` -- the prompt carries the change map, never the diff;
* ``huge`` -- past :data:`MAX_LISTED_FILES` files or :data:`HUGE_LINES` changed lines: the prompt carries the
  DIRECTORY-level map and the range's commit messages (subjects and bodies), never the diff.

Every tier may still ask (below), so a small change's context and a huge one's files can be read. The headless
run (:mod:`ai_lab.review.run`) therefore sends:

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
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Final

__all__ = [
    'DIFF_LINE_BYTES', 'HUGE_LINES', 'INLINE_BYTES', 'LOG_BYTES', 'MAX_LISTED_FILES', 'REQUESTS', 'TIERS', 'TURNS',
    'TURN_BYTES',
    'Changed', 'brief', 'change_map', 'commit_messages', 'diff', 'git', 'numstat', 'serve', 'size', 'tier',
]  # fmt: skip

#: How many turns a reviewer has: every one but the last may ask; the last must answer with findings.
TURNS: Final = 6

#: INITIAL value, to be fitted from the tier ledger. Past this many changed files the change is ``huge`` and its
#: map lists directories only (``list`` opens one): a file list that long is a page of names, not a structure.
MAX_LISTED_FILES: Final = 150

#: What one turn's answers may total; past it the rest is cut and the cut is said.
TURN_BYTES: Final = 120_000

#: INITIAL value, to be fitted from the tier ledger (:mod:`ai_lab.review.tiers`). A diff of at most this many
#: UTF-8 BYTES is inlined whole (``small``): half of one turn's answers, so the whole
#: change plus the context it asks for still fits one turn -- reading it whole is cheaper and surer than exploring.
INLINE_BYTES: Final = TURN_BYTES // 2

#: The UTF-8 BYTES one diff line averages (code plus its ``+``/``-`` and the hunk headers), turning a line count
#: into the bytes reading it costs. Every tier bound and every budget here is in UTF-8 bytes: one unit.
DIFF_LINE_BYTES: Final = 40

#: INITIAL value, to be fitted from the tier ledger. Past this many changed LINES (added plus deleted) the change
#: is ``huge``: more than every asking turn together
#: could read, so a file-level map only promises what the reviewer cannot keep.
HUGE_LINES: Final = (TURNS - 1) * TURN_BYTES // DIFF_LINE_BYTES

#: How much of a huge change's commit log its brief carries; past it the log is cut and the cut is said.
LOG_BYTES: Final = TURN_BYTES // 2

#: The tiers, smallest first; each is the headless ``diff`` context's manifest mode of the same name.
TIERS: Final = ('small', 'medium', 'huge')

#: What a reviewer may ask for, each a list of strings, and what each returns.
REQUESTS: Final = {
    'diff': 'the change to these paths in the range',
    'show': 'these files as they are at the head',
    'grep': 'git grep -n of these patterns over the head',
    'list': 'the changed files under these directories, with their line counts',
}

#: One changed file: lines added, lines deleted (``None`` for a binary file), its path at the head.
Changed = tuple[int | None, int | None, str]


def git(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    """``git -C repo *args``, never raising on git's exit; ``ValueError`` when git is not on PATH."""
    exe = shutil.which('git')
    if exe is None:
        msg = 'git is not on PATH: a review reads its change with git'
        raise ValueError(msg)
    return subprocess.run(
        [exe, '-C', repo, *args], capture_output=True, text=True, encoding='utf-8', errors='replace', check=False
    )


def numstat(range_: str, repo: str, paths: Sequence[str] = ()) -> list[Changed]:
    """The files *range_* changes (restricted to *paths* when any), renames followed; ``ValueError`` naming git's."""
    done = git(repo, 'diff', '--numstat', '-M', '--end-of-options', range_, *(['--', *paths] if paths else []))
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
        lines += ['', f'{len(rows)} files are mapped by directory only: ask `list` for a directory to see its files.']
    else:
        lines += ['', 'Files:'] + [f'  {_counts(a, d):>14}  {path}' for a, d, path in rows]
    return '\n'.join(lines)


def diff(range_: str, repo: str, paths: Sequence[str] = ()) -> str:
    """The diff of *range_* (restricted to *paths* when any), renames followed; ``ValueError`` naming git's."""
    done = git(repo, 'diff', '-M', '--end-of-options', range_, *(['--', *paths] if paths else []))
    if done.returncode:
        msg = f'git diff {range_} in {repo} failed: {done.stderr.strip()}'
        raise ValueError(msg)
    return done.stdout


def commit_messages(range_: str, repo: str) -> str:
    """The commits of *range_*, each its short sha, subject and body; ``ValueError`` naming git's."""
    done = git(repo, 'log', '--format=commit %h%n%B', '--end-of-options', range_)
    if done.returncode:
        msg = f'git log {range_} in {repo} failed: {done.stderr.strip()}'
        raise ValueError(msg)
    return done.stdout.strip()


def _huge(rows: Sequence[Changed]) -> bool:
    return len(rows) > MAX_LISTED_FILES or sum((a or 0) + (d or 0) for a, d, _ in rows) > HUGE_LINES


def tier(rows: Sequence[Changed], diff_bytes: int) -> str:
    """The tier (:data:`TIERS`) of a change of *rows* whose diff is *diff_bytes* UTF-8 bytes long."""
    if _huge(rows):
        return 'huge'
    return 'small' if diff_bytes <= INLINE_BYTES else 'medium'


def brief(range_: str, rows: Sequence[Changed], *, read_diff: Callable[[], str], read_log: Callable[[], str]) -> dict:
    """The manifest fields of the change's tier, the one place each tier's prompt payload is built.

    ``mode`` (the tier, ``empty`` for no change), ``manifest``, ``diff``, ``commits``, and ``diff_bytes`` (``None``
    when huge: its diff is never read). *read_diff* is called only below huge, *read_log* only when huge.
    """
    if not rows:
        return {'mode': 'empty', 'manifest': '', 'diff': '', 'commits': '', 'diff_bytes': 0}
    if _huge(rows):
        return {'mode': 'huge', 'manifest': change_map(range_, rows, max_files=0), 'diff': '',
                'commits': _capped(read_log()), 'diff_bytes': None}  # fmt: skip
    text = read_diff()
    mode = tier(rows, size(text))
    return {'mode': mode, 'manifest': change_map(range_, rows), 'diff': text if mode == 'small' else '',
            'commits': '', 'diff_bytes': size(text)}  # fmt: skip


def size(text: str) -> int:
    """*text*'s size in UTF-8 bytes: the ONE unit every tier bound and budget is measured in."""
    return len(text.encode('utf-8'))


def _cut(text: str, limit: int) -> str:
    """The longest prefix of *text* within *limit* UTF-8 bytes, never splitting a character."""
    return text.encode('utf-8')[:limit].decode('utf-8', errors='ignore')


def _capped(log: str) -> str:
    """A huge change's commit log, cut to :data:`LOG_BYTES` with the cut said."""
    if size(log) <= LOG_BYTES:
        return log
    return (
        f'{_cut(log, LOG_BYTES)}\n(commit log cut at {LOG_BYTES} of {size(log)} bytes; ask `show` or `diff` for more)'
    )


def _head(range_: str) -> str:
    """The revision a range ends at: ``b`` of ``a..b`` / ``a...b``, else the range itself."""
    return range_.rsplit('..', 1)[-1].lstrip('.') or 'HEAD'


def _answer(kind: str, item: str, range_: str, repo: str, rows: Sequence[Changed]) -> str:
    if kind == 'list':
        prefix = item.strip('/')
        inside = [row for row in rows if prefix in {'', '.'} or row[2].startswith(prefix + '/')]
        return '\n'.join(f'{_counts(a, d):>14}  {path}' for a, d, path in inside) or '(no changed file under it)'
    if kind == 'diff':
        done = git(repo, 'diff', '-M', '--end-of-options', range_, '--', item)
    elif kind == 'show':
        done = git(repo, 'show', f'{_head(range_)}:{item}')
    else:
        done = git(repo, 'grep', '-n', '-I', '-e', item, _head(range_))
    if done.returncode and not done.stdout:
        return f'(nothing: {done.stderr.strip() or "no match"})'
    return done.stdout or '(empty)'


def _listed(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def serve(
    requests: Mapping[str, object], *, range_: str, repo: str, rows: Sequence[Changed], budget: int = TURN_BYTES
) -> str:
    """The answers to one turn's *requests*, in :data:`REQUESTS` order, together at most *budget* UTF-8 bytes.

    An answer past what remains is CUT to it, never dropped (a file larger than a turn is still read in part), and
    the cut names how much was shown and every request left unanswered, so the reviewer can narrow and ask again.
    """
    asked = [(kind, str(item)) for kind in REQUESTS for item in _listed(requests.get(kind))]
    parts, used = [], 0
    for at, (kind, item) in enumerate(asked):
        text = f'### {kind} {item}\n{_answer(kind, item, range_, repo, rows)}\n'
        if used + size(text) > budget:
            shown = max(budget - used, 0)
            left = ', '.join(f'{k} {i}' for k, i in asked[at + 1 :]) or 'none'
            parts.append(_cut(text, shown))
            parts.append(f'\n### cut: {kind} {item} shown to {shown} of {size(text)} bytes (this turn holds '
                         f'{budget}); not answered: {left}. Narrow it, or ask the rest, next turn.\n')  # fmt: skip
            return ''.join(parts)
        parts.append(text)
        used += size(text)
    unknown = sorted(set(requests) - set(REQUESTS))
    if unknown:
        parts.append(f'### refused: {unknown} are not requests; ask only {sorted(REQUESTS)}\n')
    return ''.join(parts) or '(no request named anything to read)'
