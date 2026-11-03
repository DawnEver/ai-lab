"""WHERE A REVIEW READS ITS CHANGE: a git range, or a review bundle -- chosen once, at the entry.

A review (:mod:`ai_lab.review.__main__`) and a judge (:mod:`ai_lab.review.judge`) read a change through one
:class:`Source`, never through git directly:

* :class:`GitSource` -- ``--range <a..b>`` in ``--repo``: every read is a ``git`` subprocess (:mod:`.explore`);
* :class:`BundleSource` -- ``--bundle <dir>``, the directory dev-lab's ``reviewbundle`` materializes on a judge host
  that has NO checkout: ``bundle.json``, ``diff.patch``, and ``base/<path>`` / ``head/<path>`` holding the exact
  bytes of only the changed files. Every read is a file read; no process is ever started.

A bundle holds the CHANGED files only. A read past them -- ``show`` of an unchanged file, ``grep`` -- answers from
what it holds and says so, and :meth:`BundleSource.head_text` raises :class:`NotHeld` rather than call an unheld
file absent. A ``binary`` file carries blob ids only: it is a changed path whose content is not held. A bundle
carries no commits: its log is empty, and the summary's intent is the memory it adds plus ``--pr-body``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

from ai_lab.review.explore import Changed, commit_messages, git, numstat
from ai_lab.review.explore import diff as git_diff
from ai_lab.review.summary import MEMORY_CHARS, MEMORY_ROOT, git_intent

__all__ = ['BUNDLE_VERSION', 'BundleSource', 'GitSource', 'NotHeld', 'Source']

#: The review bundle format this module reads (dev-lab ``reviewbundle.VERSION``).
BUNDLE_VERSION: Final = 1

_NO_MATCH: Final = '(nothing: no match)'

#: The printable ASCII bytes git leaves unquoted in a diff header: ``[_PRINTABLE, _DEL)``.
_PRINTABLE: Final = 0x20
_DEL: Final = 0x7F


class NotHeld(LookupError):
    """The source cannot tell what the head holds at a path: a bundle holds only the changed files' bytes."""


class Source(Protocol):
    """What a review reads of one change; the ONE seam between the review and where the change lives."""

    label: str  # the range a prompt names, ``base..head``

    def numstat(self, paths: Sequence[str] = ()) -> list[Changed]:
        """The changed files (under *paths* when any): added and deleted lines, ``None`` for a binary file."""

    def diff(self, paths: Sequence[str] = ()) -> str:
        """The unified diff of the change (restricted to *paths* when any)."""

    def log(self) -> str:
        """The change's commit messages; empty when the source carries none."""

    def intent(self) -> dict | None:
        """The summary's intent (:func:`~ai_lab.review.summary.git_intent`'s shape); ``None``: no summary."""

    def answer(self, kind: str, item: str) -> str:
        """The answer to one ``diff`` / ``show`` / ``grep`` request (:data:`~ai_lab.review.explore.REQUESTS`)."""

    def head_text(self, path: str) -> str | None:
        """*path* at the head, ``None`` when the head has no such file; :class:`NotHeld` when it cannot say."""

    def identity(self) -> dict:
        """``{repo, sha, commits}`` for the tier ledger."""


def _head(range_: str) -> str:
    """The revision a range ends at: ``b`` of ``a..b`` / ``a...b``, else the range itself."""
    return range_.rsplit('..', 1)[-1].lstrip('.') or 'HEAD'


@dataclass(frozen=True)
class GitSource:
    """*range_* in the repository *repo*, read by git; each reader is the git one unless a test injects its own."""

    label: str
    repo: str = '.'
    read_numstat: Callable[..., list[Changed]] = numstat
    read_diff: Callable[..., str] = git_diff
    read_log: Callable[[str, str], str] = commit_messages
    read_intent: Callable[[str, str], dict] | None = git_intent

    def numstat(self, paths: Sequence[str] = ()) -> list[Changed]:  # noqa: D102 -- Source.numstat
        return self.read_numstat(self.label, self.repo, tuple(paths))

    def diff(self, paths: Sequence[str] = ()) -> str:  # noqa: D102 -- Source.diff
        return self.read_diff(self.label, self.repo, tuple(paths))

    def log(self) -> str:  # noqa: D102 -- Source.log
        return self.read_log(self.label, self.repo)

    def intent(self) -> dict | None:
        """The range's intent (:func:`~ai_lab.review.summary.git_intent`), ``None`` when no reader is injected."""
        return None if self.read_intent is None else self.read_intent(self.label, self.repo)

    def answer(self, kind: str, item: str) -> str:
        """The answer to one ``diff`` / ``show`` / ``grep`` request, as git gives it."""
        if kind == 'diff':
            done = git(self.repo, 'diff', '-M', '--end-of-options', self.label, '--', item)
        elif kind == 'show':
            done = git(self.repo, 'show', f'{_head(self.label)}:{item}')
        else:
            done = git(self.repo, 'grep', '-n', '-I', '-e', item, _head(self.label))
        if done.returncode and not done.stdout:
            return f'(nothing: {done.stderr.strip() or "no match"})'
        return done.stdout or '(empty)'

    def head_text(self, path: str) -> str | None:
        """*path* at the head, ``None`` when the head has no such file."""
        done = git(self.repo, 'show', f'{_head(self.label)}:{path}')
        return done.stdout if done.returncode == 0 else None

    def _read(self, *args: str) -> str | None:
        try:
            done = git(self.repo, *args)
        except ValueError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    def identity(self) -> dict:
        """``{repo, sha, commits}`` for the tier ledger; a part git cannot answer is ``None`` (the name falls back)."""
        top = self._read('rev-parse', '--show-toplevel') or str(Path(self.repo).resolve())
        commits = self._read('rev-list', '--count', '--end-of-options', self.label)
        return {
            'repo': Path(top).name,
            'sha': self._read('rev-parse', '--verify', '--end-of-options', _head(self.label)),
            'commits': int(commits) if commits and commits.isdigit() else None,
        }


def _quoted(path: str) -> str:
    """*path* as git's diff header quotes it (``core.quotePath``), or as is when it needs no quoting."""
    raw = path.encode('utf-8')
    if not any(not _PRINTABLE <= b < _DEL or b in b'"\\' for b in raw):
        return path
    named = {0x07: 'a', 0x08: 'b', 0x09: 't', 0x0A: 'n', 0x0B: 'v', 0x0C: 'f', 0x0D: 'r', 0x22: '"', 0x5C: '\\'}
    out = ''.join(
        f'\\{named[b]}' if b in named else f'\\{b:03o}' if not _PRINTABLE <= b < _DEL else chr(b) for b in raw
    )  # fmt: skip
    return f'"{out}"'


def _header(path: str) -> str:
    a, b = 'a/' + path, 'b/' + path
    qa, qb = _quoted(a), _quoted(b)
    return f'diff --git {qa} {qb}'


_INDEX: Final = re.compile(r'^index ([0-9a-f]{7})[0-9a-f]*\.\.([0-9a-f]{7})[0-9a-f]*', re.MULTILINE)


def _chunks(patch: str) -> dict[str, str]:
    """Each file's part of a ``git diff`` *patch*, keyed by its header line; full blob ids shortened to 7."""
    parts = re.split(r'^(?=diff --git )', patch, flags=re.MULTILINE)
    return {part.split('\n', 1)[0]: _INDEX.sub(r'index \1..\2', part) for part in parts if part.startswith('diff ')}


def _counts(chunk: str) -> tuple[int, int]:
    added = deleted = 0
    in_hunk = False
    for line in chunk.splitlines():
        if line.startswith('@@'):
            in_hunk = True
        elif in_hunk and line.startswith('+'):
            added += 1
        elif in_hunk and line.startswith('-'):
            deleted += 1
    return added, deleted


def _under(path: str, specs: Sequence[str]) -> bool:
    """Whether *path* is one of *specs* or under one (git's literal pathspec); no specs is every path."""
    tops = [spec.strip('/') for spec in specs]
    return not tops or any(top in {'', '.'} or path == top or path.startswith(top + '/') for top in tops)


def _search(pattern: str) -> Callable[[str], bool]:
    try:
        found = re.compile(pattern)
    except re.error:
        return lambda line: pattern in line
    return lambda line: found.search(line) is not None


@dataclass(frozen=True)
class _File:
    path: str
    binary: bool
    base: bytes | None
    head: bytes | None
    chunk: str


@dataclass(frozen=True)
class BundleSource:
    """The review bundle materialized at *root* (the module docstring); every read is a file read."""

    root: Path
    label: str = field(init=False)
    _doc: dict = field(init=False, repr=False)
    _files: tuple[_File, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: D105 -- the class docstring
        root = Path(self.root)
        try:
            doc = json.loads((root / 'bundle.json').read_text(encoding='utf-8'))
        except json.JSONDecodeError as bad:
            msg = f'{root / "bundle.json"} is not JSON: {bad}'
            raise ValueError(msg) from None
        if not isinstance(doc, dict) or doc.get('version') != BUNDLE_VERSION:
            msg = f'{root / "bundle.json"} is not a version-{BUNDLE_VERSION} review bundle'
            raise ValueError(msg)
        try:
            self._load(root, doc)
        except (KeyError, TypeError) as bad:
            msg = f'{root / "bundle.json"} is not a review bundle: {bad!r}'
            raise ValueError(msg) from None

    def _load(self, root: Path, doc: dict) -> None:
        patch = (root / 'diff.patch').read_text(encoding='utf-8')
        chunks = _chunks(patch)
        files = []
        for row in doc['files']:
            path = str(row['path'])
            sides = {side: self._side(root, side, path, row) for side in ('base', 'head')}
            files.append(_File(path, bool(row.get('binary')), sides['base'], sides['head'],
                               chunks.get(_header(path), '')))  # fmt: skip
        object.__setattr__(self, 'root', root)
        object.__setattr__(self, '_doc', doc)
        object.__setattr__(self, '_files', tuple(sorted(files, key=lambda one: one.path.encode('utf-8'))))
        object.__setattr__(self, 'label', f'{doc["base_sha"]}..{doc["head_sha"]}')

    @staticmethod
    def _side(root: Path, side: str, path: str, row: dict) -> bytes | None:
        """The *side*'s bytes of *path*: ``None`` where that side has no file, or the file is binary."""
        if row.get('binary') or row.get(side) is None:
            return None
        where = (root / side / path).resolve()
        if not where.is_relative_to((root / side).resolve()):
            msg = f'bundle path {path!r} leaves {root / side}'
            raise ValueError(msg)
        return where.read_bytes()

    def _held(self, paths: Sequence[str] = ()) -> list[_File]:
        return [one for one in self._files if _under(one.path, paths)]

    def numstat(self, paths: Sequence[str] = ()) -> list[Changed]:  # noqa: D102 -- Source.numstat
        return [(None, None, one.path) if one.binary else (*_counts(one.chunk), one.path) for one in self._held(paths)]

    def diff(self, paths: Sequence[str] = ()) -> str:  # noqa: D102 -- Source.diff
        return ''.join(one.chunk for one in self._held(paths))

    def log(self) -> str:
        """A bundle carries no commits."""
        return ''

    def intent(self) -> dict:
        """No commits; the memory documents the change adds or modifies, read at the head."""
        memory = [{'path': one.path, 'text': one.head.decode('utf-8', 'replace')[:MEMORY_CHARS]}
                  for one in self._files if one.path.startswith(MEMORY_ROOT) and one.head is not None]  # fmt: skip
        return {'commits': [], 'resolutions': [], 'merge_audit': [], 'memory': memory}

    def head_text(self, path: str) -> str | None:
        """*path* at the head; ``None`` when the change deleted it; :class:`NotHeld` when the bundle cannot say."""
        for one in self._files:
            if one.path == path:
                if one.binary:
                    msg = f'{path} is binary: the review bundle holds its blob id only'
                    raise NotHeld(msg)
                return None if one.head is None else one.head.decode('utf-8', 'replace')
        msg = f'{path} is not in the review bundle: it holds only the changed files'
        raise NotHeld(msg)

    def answer(self, kind: str, item: str) -> str:
        """The answer to one ``diff`` / ``show`` / ``grep`` request, from the bundle's files alone."""
        if kind == 'diff':
            return self.diff((item,)) or '(empty)'
        if kind == 'show':
            try:
                text = self.head_text(item)
            except NotHeld as unheld:
                return f'(nothing: {unheld})'
            return f'(nothing: {item} does not exist at the head)' if text is None else text or '(empty)'
        hit, head = _search(item), self._doc['head_sha']
        held = [(one.path, one.head.decode('utf-8', 'replace')) for one in self._files if one.head and not one.binary]
        lines = [f'{head}:{path}:{n}:{line}' for path, text in held
                 for n, line in enumerate(text.splitlines(), 1) if hit(line)]  # fmt: skip
        return '\n'.join(lines) + '\n' if lines else _NO_MATCH

    def identity(self) -> dict:
        """``{repo, sha, commits}`` for the tier ledger: the bundle's repository and head; commits unknown."""
        return {'repo': str(self._doc['repo']).rsplit('/', 1)[-1], 'sha': self._doc['head_sha'], 'commits': None}
