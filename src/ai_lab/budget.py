"""Daily token budgets, scoped by PROVIDER and by API KEY: a NUMBER of tokens, never dollars.

The caps are the ``[ai_lab.daily_token_cap]`` table of the per-machine file (:mod:`lab_commons.config`)::

    [ai_lab.daily_token_cap]
    default = 2000000                       # every provider the table does not name
    anthropic = 5000000                     # one provider, every key of it
    "anthropic/3f2a9c1b7d4e" = 1000000      # one key: <provider>/<fingerprint>

A call draws on its provider's cap (named, else ``default``, else :data:`DEFAULT_DAILY_TOKEN_CAP`) and, when
the table names it, on its key's cap. A key is named by its FINGERPRINT (:func:`fingerprint`, a sha256
prefix; ``python -m ai_lab.auth status`` prints it), never by its secret and never by the credential
username, so the same key is one key on every host and two usernames holding one key are one key. The tokens a day has used are a ledger,
one JSON line per call ``{"provider", "key", "tokens"}``, appended to ``tokens-<UTC date>.jsonl`` under
:func:`state_dir` -- machine-local, never tracked, so every checkout of one box draws on one budget. The
ledger is a fast LOCAL stop: a key held on several hosts is metered per host here, and the key's global
truth is its provider's own usage report (user ruling 2026-10-24, option A). A call
that would START with a cap reached raises :class:`~ai_lab.errors.DailyTokenCapReached` naming the scope and
is never sent; a call under way may finish past it, so ``max_tokens`` is clamped to what is left.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from lab_commons.config import section
from platformdirs import user_state_path

from ai_lab.errors import DailyTokenCapReached

__all__ = ['DEFAULT_DAILY_TOKEN_CAP', 'charge', 'fingerprint', 'key_scope', 'ledger_file', 'remaining', 'state_dir', 'used_today']

APP: Final = 'ai_lab'
TABLE: Final = 'daily_token_cap'
#: Two million tokens a day per provider: about 200 calls at a full 8192-token reply over a ~2k-token prompt,
#: a hard stop on a runaway loop rather than a squeeze on a normal day of review and decision calls.
DEFAULT_DAILY_TOKEN_CAP: Final = 2_000_000


#: Hex digits of a key's sha256: enough to tell keys apart, never enough to be one.
FINGERPRINT_CHARS: Final = 12


def fingerprint(secret: str) -> str:
    """A key's identity everywhere it is named: a sha256 prefix, never the key."""
    return hashlib.sha256(secret.encode()).hexdigest()[:FINGERPRINT_CHARS]


def key_scope(provider: str, key: str) -> str:
    """The table key of one API key: ``<provider>/<fingerprint>``."""
    return f'{provider}/{key}'


def _caps() -> dict[str, int]:
    table = section(APP).get(TABLE, {})
    if not isinstance(table, dict):
        msg = f'[{APP}] {TABLE} is a table of scope = tokens, got {table!r}'
        raise ValueError(msg)  # noqa: TRY004 -- a malformed machine file is a value error, not a type error
    for scope, cap in table.items():
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
            msg = f'[{APP}.{TABLE}] {scope} must be a non-negative integer number of tokens, got {cap!r}'
            raise ValueError(msg)
    return table


def state_dir() -> Path:
    """``$AI_LAB_HOME/state`` when set, else the platform's per-user local state directory."""
    home = os.environ.get('AI_LAB_HOME')
    return Path(home).expanduser() / 'state' if home else user_state_path(APP, appauthor=False)


def ledger_file() -> Path:
    """Today's (UTC) token ledger."""
    return state_dir() / f'tokens-{datetime.now(UTC).date().isoformat()}.jsonl'


def used_today(provider: str, key: str | None = None) -> int:
    """The tokens today's calls to *provider* (through *key*, when given) have reported."""
    path = ledger_file()
    rows = [json.loads(line) for line in path.read_text('utf-8').splitlines() if line] if path.is_file() else []
    return sum(r['tokens'] for r in rows if r['provider'] == provider and (key is None or r['key'] == key))


def remaining(provider: str, key: str | None = None) -> int:
    """The tokens left to a call on *provider* through *key*; :class:`DailyTokenCapReached` naming the scope."""
    caps = _caps()
    scopes = [(provider, caps.get(provider, caps.get('default', DEFAULT_DAILY_TOKEN_CAP)), None)]
    if key is not None and (scope := key_scope(provider, key)) in caps:
        scopes.append((scope, caps[scope], key))
    left = []
    for scope, cap, by_key in scopes:
        used = used_today(provider, by_key)
        if used >= cap:
            msg = (
                f'{scope} has used {used} of {cap} tokens today (UTC); no call starts there until tomorrow, '
                f'or raise [{APP}.{TABLE}] {scope!r} in the machine file'
            )
            raise DailyTokenCapReached(msg, scope=scope)
        left.append(cap - used)
    return min(left)


def charge(provider: str, key: str | None, tokens: int) -> None:
    """Append one call's *tokens* on *provider* through *key* to today's ledger."""
    path = ledger_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps({'provider': provider, 'key': key, 'tokens': tokens}) + '\n')
