"""The two verbs. A :class:`~ai_lab.client.Client`, a :class:`~ai_lab.ledger.Recorded` client and
:class:`~ai_lab.rules.Rules` all satisfy them structurally, so a caller types against these."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_lab.answers import Decision, Response
from ai_lab.spec import DecisionRequest, ResponseRequest

__all__ = ['Decider', 'Responder']


@runtime_checkable
class Decider(Protocol):
    name: str

    def decide(self, request: DecisionRequest) -> Decision: ...


@runtime_checkable
class Responder(Protocol):
    name: str

    def respond(self, request: ResponseRequest) -> Response: ...
