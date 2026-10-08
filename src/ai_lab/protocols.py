"""The two verbs every model client offers.

A :class:`~ai_lab.client.Client`, a :class:`~ai_lab.ledger.Recorded` client and
:class:`~ai_lab.rules.Rules` all satisfy them structurally, so a caller types against these.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_lab.answers import Decision, Response
from ai_lab.spec import DecisionRequest, ResponseRequest

__all__ = ['Decider', 'Responder']


@runtime_checkable
class Decider(Protocol):
    """Anything that answers closed questions."""

    name: str

    def decide(self, request: DecisionRequest) -> Decision:
        """Answer every question of ``request``."""


@runtime_checkable
class Responder(Protocol):
    """Anything that answers in a declared JSON schema."""

    name: str

    def respond(self, request: ResponseRequest) -> Response:
        """An object of ``request``'s schema."""
