"""``python -m ai_lab.auth login|status`` -- model API keys live in the git credential store, and only there.

ONE MECHANISM ON EVERY PLATFORM (user ruling 2026-10-13). A key is stored with ``git credential
approve`` under the ``https`` host of the provider's endpoint, exactly as the forge token is; the helper
git is configured with decides where it persists (Git Credential Manager on Windows and macOS, the
keychain or ``credential-store`` on Linux). No environment variable is read, so there is no second
place for a key to hide and no platform-specific ``setx``/``export`` recipe.

ONE FIELD, TWO JOBS, NO SECOND COPY. A provider row's ``key_name`` (``OPENAI_API_KEY``) says both THAT
it needs a key (non-empty) and what the key is conventionally called -- which is all ``--from-file``
needs to route a ``NAME value`` line to a provider. A separate ``keyed = true`` flag beside it could
disagree with it; deriving the flag from the name cannot. Providers sharing a host share one stored key.

SLOTS ARE FALLBACK KEYS OF ONE ACCOUNT. ``login <provider> --slot N`` stores username ``api-key-N``
(slot 1, the default, is ``api-key``); a call falls over to the next slot on an account-level failure
(:mod:`ai_lab.client`). ``status`` counts each provider's slots up to the first gap.

A TASK OWNS ITS KEYS. ``login <provider> --task <task> [-s N]`` stores username ``api-key-<task>[-N]``
under the same host (:func:`lab_commons.credentials.key_username`, the one rule; the tasks are declared in
``API_KEY_TASKS``): the slots ``connect(spec, task=...)`` walks, and nothing else does. ``-s``, ``--slot``
and ``--seat`` are one argument. ``status`` lists, per provider, each task's stored slot count.

EVERY TASK GETS ITS OWN KEY (user ruling 2026-10-19). ``login`` refuses a key that another credential of
the same host already holds -- another task, another slot of the same task, or the default (which counts
as its own task) -- naming that credential and the key's fingerprint (a sha256 prefix; never the key).
Re-storing the identical key into the same credential is a no-op. ``status`` lists every group of
credentials sharing one key. The stored keys are read through :func:`lab_commons.credentials.api_keys`.

The key never travels in argv: ``login <provider>`` reads it from the terminal without echo, or from
stdin when piped; ``--from-file -`` reads a key file from stdin (``ssh box python -m ai_lab.auth login
--from-file - < keys``). ``--task`` and ``--slot`` apply to a key file exactly as to a prompted key, and
``login <provider> --from-file`` imports only that provider's line.
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from lab_commons.credentials import API_KEY_TASKS, NoCredential, api_keys, key_username, store_credential

from ai_lab.budget import fingerprint
from ai_lab.client import Provider, catalogue, stored_slots

__all__ = ['main', 'parse_key_file']

#: ``[export ]NAME<= or whitespace>value``: the name, then the first ``=`` or run of whitespace.
_LINE = re.compile(r'^(?:export\s+)?([^=\s]+)\s*(?:=|\s)\s*(.*)$')

_QUOTES = ('"', "'")


def parse_key_file(text: str) -> tuple[list[tuple[str, str]], list[int]]:
    """``(NAME, value)`` pairs from a key file, and the line numbers that held no value.

    Accepts ``NAME=value``, ``export NAME=value`` and ``NAME value``; strips CR, a BOM and one pair
    of surrounding quotes; skips blank and ``#`` lines.
    """
    pairs, malformed = [], []
    for number, raw in enumerate(text.lstrip('﻿').splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = _LINE.match(line)
        value = match.group(2).strip() if match else ''
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:  # noqa: PLR2004 -- a quote pair
            value = value[1:-1]
        if not value:
            malformed.append(number)
            continue
        pairs.append((match.group(1), value))
    return pairs, malformed


def _keyed() -> dict[str, Provider]:
    return {name: row for name, row in catalogue().items() if row.keyed}


def _held(host: str) -> dict[str, str]:
    """``{username: key}`` for every key *host* holds: the default's slots, then each task's."""
    held = {}
    for task in (None, *sorted(API_KEY_TASKS)):
        try:
            keys = api_keys(host, task=task)
        except NoCredential:
            continue
        held |= {key_username(task, slot): key for slot, key in enumerate(keys, start=1)}
    return held


def _store(host: str, secret: str, username: str) -> bool | None:
    """Store *secret* as *username*; ``None`` when that credential already holds it, ``False`` on refusal."""
    held = _held(host)
    if held.get(username) == secret:
        return None
    if clash := sorted(other for other, key in held.items() if key == secret):
        sys.stderr.write(
            f'refused: {host} already holds this key (sha256 {fingerprint(secret)}...) as '
            f'{", ".join(map(repr, clash))}; every task and slot gets its own key\n'
        )
        return False
    try:
        store_credential(host, username, secret)
    except NoCredential as failed:
        sys.stderr.write(f'{failed}\n')
        return False
    return True


def _not_keyed(name: str | None, keyed: dict[str, Provider]) -> bool:
    """Whether *name* was given and is no keyed provider, saying why."""
    if name is None or name in keyed:
        return False
    table = catalogue()
    sys.stderr.write(
        f'{name} needs no key\n' if name in table else f'unknown provider {name!r}; known: {sorted(table)}\n'
    )
    return True


def _read_pairs(source: str | None, provider: Provider | None, which: str) -> list[tuple[str, str]] | None:
    """``(NAME, key)`` pairs from the key file *source*, else *provider*'s one key read without echo."""
    if source is not None:
        text = sys.stdin.read() if source == '-' else Path(source).read_text(encoding='utf-8')
        pairs, malformed = parse_key_file(text)
        for number in malformed:
            sys.stderr.write(f'line {number}: no value; skipped\n')
        return pairs
    prompt = f'{provider.key_name} for {provider.host} ({which}): '
    secret = (getpass.getpass(prompt) if sys.stdin.isatty() else sys.stdin.readline()).strip()
    if not secret:
        sys.stderr.write('empty key; nothing stored\n')
        return None
    return [(provider.key_name, secret)]


def _login(name: str | None, source: str | None, slot: int, task: str | None) -> int:
    """Store keys as ``key_username(task, slot)``: *name*'s key from the terminal/stdin, or a key file's.

    *source* (a path, ``-`` for stdin) supplies ``NAME value`` lines; *name* alone prompts for its key,
    and *name* with *source* imports only that provider's line. One path, so ``--task``/``--slot`` mean
    the same thing whichever way the key arrives.
    """
    try:
        username = key_username(task, slot)
    except ValueError as refused:
        sys.stderr.write(f'{refused}\n')
        return 1
    keyed = _keyed()
    if _not_keyed(name, keyed):
        return 1
    which = f'slot {slot}' if task is None else f'task {task} slot {slot}'
    pairs = _read_pairs(source, keyed[name] if source is None else None, which)
    if pairs is None:
        return 1
    by_name: dict[str, list[Provider]] = {}
    for row in keyed.values() if name is None else [keyed[name]]:
        by_name.setdefault(row.key_name, []).append(row)
    scope = '' if name is None else f'{name!r} '
    stored = 0
    for key_name, secret in pairs:
        if key_name not in by_name:
            sys.stderr.write(f'{key_name}: no provider {scope}names this key; skipped\n')
            continue
        for host in sorted({row.host for row in by_name[key_name]}):
            outcome = _store(host, secret, username)
            if outcome is False:
                return 1
            sharing = sorted(n for n, row in keyed.items() if row.host == host)
            sys.stdout.write(
                f'{host} already holds this key as {username!r} ({which}); nothing changed\n'
                if outcome is None
                else f'{key_name}: stored key {which} for {host} ({", ".join(sharing)})\n'
            )
            stored += 1
    if not stored:
        sys.stderr.write('no key in the file matched a provider; nothing stored\n')
    return 0 if stored else 1


def _status_lines(rows: Iterable[Provider]) -> list[str]:
    lines = []
    for row in rows:
        if not row.keyed:
            lines.append(f'{row.name:20} {row.auth:13} {"-":45} - -')
            continue
        held = {task: stored_slots(row.host, task) for task in sorted(API_KEY_TASKS)}
        tasks = ','.join(f'{task}:{count}' for task, count in held.items() if count) or '-'
        lines.append(f'{row.name:20} {row.auth:13} {row.host:45} {stored_slots(row.host)} {tasks}')
    return lines


def _shared_lines(rows: Iterable[Provider]) -> list[str]:
    """One line per group of credentials of one host that hold the same key."""
    lines = []
    for host in sorted({row.host for row in rows if row.keyed}):
        groups: dict[str, list[str]] = {}
        for username, key in _held(host).items():
            groups.setdefault(key, []).append(username)
        lines += [
            f'shared key sha256 {fingerprint(key)}... on {host}: {", ".join(map(repr, names))}'
            for key, names in groups.items()
            if len(names) > 1
        ]
    if lines:
        lines.append(
            'each task needs its own key; a default<->task pair is expected after a migration '
            'and is resolved by deleting the default copy'
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``login <provider>``, ``login --from-file PATH|-`` or ``status``."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.auth', description=__doc__.split('\n', 1)[0])
    verbs = parser.add_subparsers(dest='verb', required=True)
    login = verbs.add_parser('login', help='store a key: read from the terminal/stdin, or import a key file')
    login.add_argument(
        'provider', nargs='?', help='a keyed provider; its key is read without echo (or its line of --from-file)'
    )
    login.add_argument('--from-file', metavar='PATH', help='a NAME=value / NAME value file; "-" reads stdin')
    login.add_argument(
        '-s', '--slot', '--seat', dest='slot', type=int, default=1,
        help='the key slot to store: 1 = api-key[-<task>], N = api-key[-<task>]-N, tried in order',
    )  # fmt: skip
    login.add_argument('--task', help=f'store a key one task owns (api-key-<task>[-N]): {sorted(API_KEY_TASKS)}')
    verbs.add_parser(
        'status', help='each provider: auth kind, its key slots, and each task:slots it holds (never a key)'
    )
    args = parser.parse_args(argv)
    if args.verb == 'status':
        rows = list(catalogue().values())
        sys.stdout.write('\n'.join(_status_lines(rows) + _shared_lines(rows)) + '\n')
        return 0
    if args.provider is None and args.from_file is None:
        login.error('give a provider, --from-file, or both')
    return _login(args.provider, args.from_file, args.slot, args.task)


if __name__ == '__main__':
    raise SystemExit(main())
