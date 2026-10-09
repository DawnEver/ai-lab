"""What came back: one typed answer per question, and where its probability came from.

``Basis`` names the provenance instead of implying it. A vendor that returns a distribution is
``NATIVE``; a model that only STATED an answer is ``STATED`` and its probabilities are ``None``,
never a 1.0 dressed up as a measurement; a deterministic rule is ``RULE``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from ai_lab.usage import Usage

__all__ = [
    'YES_AT',
    'Answer',
    'Basis',
    'ChoiceAnswer',
    'Decision',
    'PredicateAnswer',
    'Refusal',
    'Response',
    'ScoreAnswer',
]


#: A native predicate reads as yes from this probability up.
YES_AT = 0.5


class Basis(StrEnum):
    """Where an answer's probability came from."""

    NATIVE = 'native'
    STATED = 'stated'
    RULE = 'rule'


@dataclass(frozen=True, slots=True)
class PredicateAnswer:
    """A yes/no answer, with the probability it is yes when the vendor gave one."""

    name: str
    value: bool
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """The picked option, with the distribution over options when the vendor gave one."""

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
    """The model declined to answer this question."""

    name: str
    reason: str = ''


Answer = PredicateAnswer | ChoiceAnswer | ScoreAnswer | Refusal


@dataclass(frozen=True, slots=True)
class Decision:
    """One answer per question, from one provider and model."""

    answers: Mapping[str, Answer]
    basis: Basis
    provider: str
    model: str
    latency_s: float = 0.0
    usage: Usage | None = None

    def __getitem__(self, name: str) -> Answer:
        """The answer to the question named ``name``."""
        return self.answers[name]


@dataclass(frozen=True, slots=True)
class Response:
    """An object of the requested schema, from one provider and model."""

    value: Mapping
    provider: str
    model: str
    latency_s: float = 0.0
    usage: Usage | None = None
