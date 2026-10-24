"""The one budget: a DAILY NUMBER OF TOKENS (input + output as each reply reports them), never dollars.

The cap is ``daily_token_cap`` in the ``[ai_lab]`` table of the per-machine file
(:mod:`lab_commons.config`), else :data:`DEFAULT_DAILY_TOKEN_CAP`. The tokens a day has used are a
ledger of one integer per call, appended to ``tokens-<UTC date>.jsonl`` under :func:`state_dir` -- a
machine-local directory, never a tracked file, so every checkout and worktree of one box draws on one
budget. A call that would START once the day's tokens reach the cap raises
:class:`~ai_lab.errors.DailyTokenCapReached` and is never sent; a call already under way may finish past
it, which is why a request's ``max_tokens`` is clamped to what the day has left (:func:`remaining`).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from lab_commons.config import section
from platformdirs import user_state_path

from ai_lab.errors import DailyTokenCapReached

__all__ = [
    'DEFAULT_DAILY_TOKEN_CAP',
    'charge',
    'daily_token_cap',
    'ledger_file',
    'remaining',
    'state_dir',
    'used_today',
]

APP: Final = 'ai_lab'
#: Two million tokens a day: about 200 calls at a full 8192-token reply over a ~2k-token prompt, an order
#: of magnitude above a measured day of review and decision calls, and a hard stop on a runaway loop.
DEFAULT_DAILY_TOKEN_CAP: Final = 2_000_000


def daily_token_cap() -> int:
    """``[ai_lab] daily_token_cap`` from the machine file, else :data:`DEFAULT_DAILY_TOKEN_CAP`."""
    cap = section(APP).get('daily_token_cap', DEFAULT_DAILY_TOKEN_CAP)
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        msg = f'[{APP}] daily_token_cap must be a non-negative integer number of tokens, got {cap!r}'
        raise ValueError(msg)
    return cap


def state_dir() -> Path:
    """``$AI_LAB_HOME/state`` when set, else the platform's per-user local state directory."""
    home = os.environ.get('AI_LAB_HOME')
    return Path(home).expanduser() / 'state' if home else user_state_path(APP, appauthor=False)


def ledger_file() -> Path:
    """Today's (UTC) token ledger."""
    return state_dir() / f'tokens-{datetime.now(UTC).date().isoformat()}.jsonl'


def used_today() -> int:
    """The tokens every call today has reported."""
    path = ledger_file()
    return sum(json.loads(line) for line in path.read_text('utf-8').splitlines() if line) if path.is_file() else 0


def remaining() -> int:
    """The tokens today has left; :class:`DailyTokenCapReached` when none, so no call starts."""
    cap, used = daily_token_cap(), used_today()
    if used >= cap:
        msg = (
            f'today (UTC) has used {used} of {cap} tokens; no call starts until tomorrow, or raise '
            f'[{APP}] daily_token_cap in the machine file'
        )
        raise DailyTokenCapReached(msg)
    return cap - used


def charge(tokens: int) -> None:
    """Append *tokens* to today's ledger."""
    path = ledger_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as out:
        out.write(f'{tokens}\n')
