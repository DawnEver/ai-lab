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

The key never travels in argv: ``login <provider>`` reads it from the terminal without echo, or from
stdin when piped; ``--from-file -`` reads a key file from stdin (``ssh box python -m ai_lab.auth login
--from-file - < keys``).
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from lab_commons.credentials import NoCredential, store_credential

from ai_lab.client import Provider, providers, slot_username, stored_slots

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
    return {name: row for name, row in providers().items() if row.keyed}


def _store(host: str, secret: str, slot: int = 1) -> bool:
    try:
        store_credential(host, slot_username(slot), secret)
    except NoCredential as failed:
        sys.stderr.write(f'{failed}\n')
        return False
    return True


def _login_one(name: str, slot: int) -> int:
    if slot < 1:
        sys.stderr.write(f'--slot is 1 or more, not {slot}\n')
        return 1
    table = providers()
    if name not in table:
        sys.stderr.write(f'unknown provider {name!r}; known: {sorted(table)}\n')
        return 1
    provider = table[name]
    if not provider.keyed:
        sys.stderr.write(f'{name} needs no key\n')
        return 1
    prompt = f'{provider.key_name} for {provider.host} (slot {slot}): '
    secret = (getpass.getpass(prompt) if sys.stdin.isatty() else sys.stdin.readline()).strip()
    if not secret:
        sys.stderr.write('empty key; nothing stored\n')
        return 1
    if not _store(provider.host, secret, slot):
        return 1
    sharing = sorted(n for n, row in _keyed().items() if row.host == provider.host)
    sys.stdout.write(f'stored key slot {slot} for {provider.host} ({", ".join(sharing)})\n')
    return 0


def _login_file(source: str) -> int:
    text = sys.stdin.read() if source == '-' else Path(source).read_text(encoding='utf-8')
    pairs, malformed = parse_key_file(text)
    for number in malformed:
        sys.stderr.write(f'line {number}: no value; skipped\n')
    by_name: dict[str, list[Provider]] = {}
    for row in _keyed().values():
        by_name.setdefault(row.key_name, []).append(row)
    stored = 0
    for name, secret in pairs:
        if name not in by_name:
            sys.stderr.write(f'{name}: no provider names this key; skipped\n')
            continue
        for host in sorted({row.host for row in by_name[name]}):
            if not _store(host, secret):
                return 1
            sharing = sorted(row.name for row in by_name[name] if row.host == host)
            sys.stdout.write(f'{name}: stored for {host} ({", ".join(sharing)})\n')
            stored += 1
    if not stored:
        sys.stderr.write('no key in the file matched a provider; nothing stored\n')
    return 0 if stored else 1


def _status_lines(rows: Iterable[Provider]) -> list[str]:
    lines = []
    for row in rows:
        if not row.keyed:
            lines.append(f'{row.name:20} {row.auth:13} {"-":45} -')
            continue
        lines.append(f'{row.name:20} {row.auth:13} {row.host:45} {stored_slots(row.host)}')
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``login <provider>``, ``login --from-file PATH|-`` or ``status``."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.auth', description=__doc__.split('\n', 1)[0])
    verbs = parser.add_subparsers(dest='verb', required=True)
    login = verbs.add_parser('login', help='store a key: read from the terminal/stdin, or import a key file')
    target = login.add_mutually_exclusive_group(required=True)
    target.add_argument('provider', nargs='?', help='a keyed provider; the key is read without echo, never argv')
    target.add_argument('--from-file', metavar='PATH', help='a NAME=value / NAME value file; "-" reads stdin')
    login.add_argument(
        '--slot', type=int, default=1, help='the fallback key slot to store (1 = api-key, N = api-key-N)'
    )
    verbs.add_parser('status', help='each provider: auth kind, and how many key slots a keyed one holds (never a key)')
    args = parser.parse_args(argv)
    if args.verb == 'status':
        sys.stdout.write('\n'.join(_status_lines(providers().values())) + '\n')
        return 0
    return _login_file(args.from_file) if args.from_file else _login_one(args.provider, args.slot)


if __name__ == '__main__':
    raise SystemExit(main())
