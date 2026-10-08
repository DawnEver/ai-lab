"""What came back: one typed answer per question, and where its probability came from.

``Basis`` names the provenance instead of implying it. A vendor that returns a distribution is
``NATIVE``; a model that only STATED an answer is ``STATED`` and its probabilities are ``None``,
never a 1.0 dressed up as a measurement; a deterministic rule is ``RULE``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    'Answer',
    'Basis',
    'ChoiceAnswer',
    'Decision',
    'PredicateAnswer',
    'Refusal',
    'Response',
    'ScoreAnswer',
]


class Basis(StrEnum):
    NATIVE = 'native'
    STATED = 'stated'
    RULE = 'rule'


@dataclass(frozen=True, slots=True)
class PredicateAnswer:
    name: str
    value: bool
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    name: str
    choice: str
    probabilities: Mapping[str, float] | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """``level`` is the most likely label; ``score`` is the expected level index (the index itself when STATED)."""

    name: str
    level: str
    score: float
    probabilities: Mapping[str, float] | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Refusal:
    name: str
    reason: str = ''


Answer = PredicateAnswer | ChoiceAnswer | ScoreAnswer | Refusal


@dataclass(frozen=True, slots=True)
class Decision:
    answers: Mapping[str, Answer]
    basis: Basis
    provider: str
    model: str
    latency_s: float = 0.0

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]


@dataclass(frozen=True, slots=True)
class Response:
    value: Mapping
    provider: str
    model: str
    latency_s: float = 0.0
