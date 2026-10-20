"""Every spelling ai-lab has retired, with its replacement and its reason -- refused by name, never aliased.

This is the ONE place a retired spelling may appear. A provider row, a ``connect`` spec or a field that
uses one is refused here naming what replaced it; nothing reads a retired spelling as its replacement.
``scanned`` marks a spelling distinctive enough that its appearance anywhere else in the tracked tree is
a leftover (``tests/architecture/test_retired_names_are_registered.py`` reds on it); a spelling that is
also an ordinary word or a live name of another kind is registered but not scanned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ai_lab.errors import Unsupported

__all__ = ['KINDS', 'RETIRED', 'Retired', 'refuse']

#: What a retired spelling named.
KINDS: Final = frozenset({'provider', 'wire', 'field', 'auth'})

#: Why the two subscription doors left (user ruling 2026-10-20): ai-lab is the pay-per-token API layer.
_API_ONLY = (
    'ai-lab is the pay-per-token API layer only; a subscription is used inside a host session '
    '(Claude Code, Codex, or fabric agent mode), never through ai-lab'
)


@dataclass(frozen=True, slots=True)
class Retired:
    """One retired spelling of one kind: what replaced it, why, and since when."""

    kind: str
    spelling: str
    replacement: str
    reason: str
    since: str
    scanned: bool = True

    def __post_init__(self) -> None:
        """Refuse an unknown kind."""
        if self.kind not in KINDS:
            msg = f'retired {self.spelling!r} names unknown kind {self.kind!r}; the kinds are {sorted(KINDS)}'
            raise ValueError(msg)


RETIRED: Final = (
    Retired(
        'field', 'key_env', 'key_name',
        'keys are no longer read from the environment; store the key with python -m ai_lab.auth login <provider>',
        '2026-10-13',
    ),
    Retired(
        'provider', 'openai_decisions', 'openai', 'one provider is one account; the model picks the wire',
        '2026-10-16', scanned=False,  # still the live name of the decisions WIRE
    ),
    Retired('provider', 'claude_code', 'anthropic', _API_ONLY, '2026-10-20'),
    Retired('provider', 'codex', 'openai', _API_ONLY, '2026-10-20', scanned=False),  # a Claude-role model key
    Retired('wire', 'claude_code', 'anthropic_messages', _API_ONLY, '2026-10-20'),
    Retired('wire', 'codex', 'openai_responses', _API_ONLY, '2026-10-20', scanned=False),
    Retired('auth', 'subscription', 'api', _API_ONLY, '2026-10-20', scanned=False),
    Retired('field', 'api_provider', '(nothing: every row is an api or local account)', _API_ONLY, '2026-10-20'),
)  # fmt: skip

_BY_NAME: Final = {(r.kind, r.spelling): r for r in RETIRED}


def refuse(kind: str, spelling: str, origin: str) -> None:
    """Raise :class:`Unsupported` naming the replacement when *spelling* is a retired *kind*; else return."""
    retired = _BY_NAME.get((kind, spelling))
    if retired is not None:
        msg = (
            f'{kind} {spelling!r} ({origin}) was retired {retired.since}: {retired.reason}; '
            f'use {retired.replacement!r} (registered in ai_lab.retired)'
        )
        raise Unsupported(msg)
