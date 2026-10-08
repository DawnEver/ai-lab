"""What is asked: the context a model reads, the questions it answers, the shape it responds in.

Everything here is declared ONCE and every vendor payload is derived from it: a wire encodes these
objects and never invents a field of its own. The question vocabulary is the one both native
decision endpoints already share -- a yes/no condition, a pick from fixed options, a rating on
ordered levels -- so a native wire is a renaming and an emulated one is a derived JSON schema.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    'Choice',
    'Context',
    'DecisionRequest',
    'Fields',
    'Image',
    'Level',
    'Option',
    'Part',
    'Predicate',
    'Question',
    'ResponseRequest',
    'Score',
    'Text',
    'canonical',
]

Scalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Text:
    """Free text: instructions, domain knowledge, a report."""

    text: str


@dataclass(frozen=True, slots=True)
class Fields:
    """Named values: the structured state a rule can read directly and a model reads as JSON."""

    values: Mapping[str, Scalar]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'values', dict(self.values))

    def as_text(self) -> str:
        return json.dumps(self.values, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Image:
    """Raw image bytes and their MIME type; a wire that cannot carry an image refuses the request."""

    data: bytes
    mime: str = 'image/png'

    def __post_init__(self) -> None:
        if not self.mime.startswith('image/'):
            raise ValueError(f'an Image needs an image/* MIME type, got {self.mime!r}')

    def data_url(self) -> str:
        return f'data:{self.mime};base64,{base64.b64encode(self.data).decode("ascii")}'


Part = Text | Fields | Image
Context = tuple[Part, ...]


@dataclass(frozen=True, slots=True)
class Option:
    value: str
    description: str = ''


@dataclass(frozen=True, slots=True)
class Level:
    label: str
    description: str = ''


@dataclass(frozen=True, slots=True)
class Predicate:
    """Is a condition true? Answered with a probability that it is."""

    name: str
    instructions: str


@dataclass(frozen=True, slots=True)
class Choice:
    """Which one of fixed options? Answered with the option (and a distribution when native)."""

    name: str
    instructions: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        values = [o.value for o in self.options]
        if len(values) < 2 or len(set(values)) != len(values):
            raise ValueError(f'choice {self.name!r} needs at least two distinct options, got {values}')


@dataclass(frozen=True, slots=True)
class Score:
    """Where on ordered levels? Level ``i`` is worth ``i``; the score is the expected index."""

    name: str
    instructions: str
    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        labels = [lv.label for lv in self.levels]
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError(f'score {self.name!r} needs at least two distinct levels, got {labels}')


Question = Predicate | Choice | Score


def _context(parts: Context) -> Context:
    parts = tuple(parts)
    for part in parts:
        if not isinstance(part, Text | Fields | Image):
            raise TypeError(f'a context part is Text, Fields or Image, got {type(part).__name__}')
    return parts


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Several independent questions about one context, answered in one call."""

    context: Context
    questions: tuple[Question, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'context', _context(self.context))
        object.__setattr__(self, 'questions', tuple(self.questions))
        names = [q.name for q in self.questions]
        if not names or len(set(names)) != len(names):
            raise ValueError(f'a decision request needs at least one question and distinct names, got {names}')

    @property
    def has_image(self) -> bool:
        return any(isinstance(p, Image) for p in self.context)


@dataclass(frozen=True, slots=True)
class ResponseRequest:
    """An answer of a declared JSON-schema shape, about a context, under instructions."""

    instructions: str
    context: Context
    schema: Mapping = field(default_factory=dict)
    name: str = 'response'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'context', _context(self.context))
        if self.schema.get('type') != 'object':
            raise ValueError('a response schema is a JSON-schema object (type: object)')

    @property
    def has_image(self) -> bool:
        return any(isinstance(p, Image) for p in self.context)


def _plain(value: object) -> object:
    if isinstance(value, Image):
        return {'image': value.mime, 'sha256': hashlib.sha256(value.data).hexdigest()}
    if isinstance(value, Text | Fields | Option | Level | Predicate | Choice | Score):
        return {'kind': type(value).__name__, **{k: _plain(getattr(value, k)) for k in value.__slots__}}
    if isinstance(value, DecisionRequest | ResponseRequest):
        return {k: _plain(getattr(value, k)) for k in value.__slots__}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def canonical(value: object) -> str:
    """A stable text for hashing: images by digest, never by content, so a key carries no pixels."""
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'))
