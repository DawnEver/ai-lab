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
from dataclasses import asdict, dataclass

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

#: A closed question with fewer answers than this decides nothing.
_AT_LEAST = 2


@dataclass(frozen=True, slots=True)
class Text:
    """Free text: instructions, domain knowledge, a report."""

    text: str


@dataclass(frozen=True, slots=True)
class Fields:
    """Named values: the structured state a rule can read directly and a model reads as JSON."""

    values: dict[str, Scalar]

    def __post_init__(self) -> None:
        """Refuse anything but a dict."""
        if not isinstance(self.values, dict):
            msg = f'Fields takes a dict of named values, got {type(self.values).__name__}'
            raise TypeError(msg)

    def as_text(self) -> str:
        """The values as sorted JSON -- what a model reads."""
        return json.dumps(self.values, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Image:
    """Raw image bytes and their MIME type; a wire that cannot carry an image refuses the request."""

    data: bytes
    mime: str = 'image/png'

    def __post_init__(self) -> None:
        """Refuse a non-image MIME type."""
        if not self.mime.startswith('image/'):
            msg = f'an Image needs an image/* MIME type, got {self.mime!r}'
            raise ValueError(msg)

    def data_url(self) -> str:
        """The image as a base64 ``data:`` URL."""
        return f'data:{self.mime};base64,{base64.b64encode(self.data).decode("ascii")}'


Part = Text | Fields | Image
Context = tuple[Part, ...]


@dataclass(frozen=True, slots=True)
class Option:
    """One option of a choice: its value and when to pick it."""

    value: str
    description: str = ''


@dataclass(frozen=True, slots=True)
class Level:
    """One level of a score: its label and what it means."""

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
        """Refuse fewer than two options, or a repeated one."""
        values = [o.value for o in self.options]
        if len(values) < _AT_LEAST or len(set(values)) != len(values):
            msg = f'choice {self.name!r} needs at least two distinct options, got {values}'
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Score:
    """Where on ordered levels? Level ``i`` is worth ``i``; the score is the expected index."""

    name: str
    instructions: str
    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        """Refuse fewer than two levels, or a repeated one."""
        labels = [lv.label for lv in self.levels]
        if len(labels) < _AT_LEAST or len(set(labels)) != len(labels):
            msg = f'score {self.name!r} needs at least two distinct levels, got {labels}'
            raise ValueError(msg)


Question = Predicate | Choice | Score


def _require_context(parts: Context) -> None:
    if not isinstance(parts, tuple):
        msg = f'a context is a tuple of parts, got {type(parts).__name__}'
        raise TypeError(msg)
    for part in parts:
        if not isinstance(part, Text | Fields | Image):
            msg = f'a context part is Text, Fields or Image, got {type(part).__name__}'
            raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Several independent questions about one context, answered in one call."""

    context: Context
    questions: tuple[Question, ...]

    def __post_init__(self) -> None:
        """Refuse a malformed context and an empty or repeated question set."""
        _require_context(self.context)
        if not isinstance(self.questions, tuple):
            msg = f'questions are a tuple, got {type(self.questions).__name__}'
            raise TypeError(msg)
        names = [q.name for q in self.questions]
        if not names or len(set(names)) != len(names):
            msg_0 = f'a decision request needs at least one question and distinct names, got {names}'
            raise ValueError(msg_0)

    @property
    def has_image(self) -> bool:
        """Whether any context part is an image."""
        return any(isinstance(p, Image) for p in self.context)


@dataclass(frozen=True, slots=True)
class ResponseRequest:
    """An answer of a declared JSON-schema shape, about a context, under instructions."""

    instructions: str
    context: Context
    schema: dict
    name: str = 'response'

    def __post_init__(self) -> None:
        """Refuse a malformed context and a schema that is not an object."""
        _require_context(self.context)
        if self.schema.get('type') != 'object':
            msg = 'a response schema is a JSON-schema object (type: object)'
            raise ValueError(msg)

    @property
    def has_image(self) -> bool:
        """Whether any context part is an image."""
        return any(isinstance(p, Image) for p in self.context)


def _digest(value: object) -> object:
    """``json.dumps``' fallback: bytes appear by digest, so a key carries no pixels."""
    if isinstance(value, bytes):
        return {'sha256': hashlib.sha256(value).hexdigest()}
    msg = f'{type(value).__name__} is not part of a request'
    raise TypeError(msg)


def canonical(request: DecisionRequest | ResponseRequest) -> str:
    """A stable text for hashing a request: every field, sorted keys, images by digest.

    Part and question kinds need no tag: each one's field set is distinct from every other's, so
    ``dataclasses.asdict`` already tells a ``Text`` from a ``Fields`` and a ``Choice`` from a ``Score``.
    """
    return json.dumps(asdict(request), sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=_digest)
