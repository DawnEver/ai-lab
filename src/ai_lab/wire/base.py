"""What a wire IS: a record of the translations it carries, with an absent verb spelled ``None``.

A wire is a PROTOCOL, never a company: each module in this package defines exactly one :class:`Wire`
named ``WIRE``, and :data:`ai_lab.wire.WIRES` is the one table of them. The verbs a provider has are
read off that record -- ``respond`` and ``decide`` are present or ``None`` -- so capability is the
declaration itself rather than something probed for at call time. The same holds for the two
optional hooks: ``parse_usage`` reads the vendor's token counts off a raw reply, and ``effort`` puts
a reasoning effort onto a call; a wire without one does not report usage or take an effort.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from ai_lab.answers import Answer
from ai_lab.spec import Context, DecisionRequest, Fields, Image, ResponseRequest, Text
from ai_lab.transport import CliCall, HttpCall
from ai_lab.usage import Usage

__all__ = ['Wire', 'body_effort', 'is_closed', 'joined_text', 'text_of']

Call = HttpCall | CliCall
#: ``(request, model, endpoint, key) -> call``: the endpoint is a base URL or a CLI binary name.
Build = Callable[[ResponseRequest, str, str, str], Call]
BuildDecision = Callable[[DecisionRequest, str, str, str], Call]


@dataclass(frozen=True, slots=True)
class Wire:
    """One protocol. ``decide`` is ``None`` unless the vendor answers typed questions natively."""

    name: str
    cli: bool
    respond: Build | None = None
    parse_response: Callable[[object], Mapping] | None = None
    decide: BuildDecision | None = None
    parse_decision: Callable[[object, DecisionRequest], dict[str, Answer]] | None = None
    parse_usage: Callable[[object], Usage | None] | None = None
    effort: Callable[[Call, str], Call] | None = None

    def __post_init__(self) -> None:
        """Refuse a half-declared verb and a wire that carries none."""
        if (self.respond is None) != (self.parse_response is None):
            msg = f'wire {self.name!r} must define respond and parse_response together'
            raise ValueError(msg)
        if (self.decide is None) != (self.parse_decision is None):
            msg = f'wire {self.name!r} must define decide and parse_decision together'
            raise ValueError(msg)
        if self.respond is None and self.decide is None:
            msg = f'wire {self.name!r} carries no verb'
            raise ValueError(msg)


def body_effort(*path: str) -> Callable[[Call, str], Call]:
    """An ``effort`` hook writing the effort at ``path`` inside an HTTP body (``('reasoning', 'effort')``)."""

    def apply(call: Call, effort: str) -> Call:
        if not isinstance(call, HttpCall):
            msg = 'a body effort applies to an HTTP call'
            raise TypeError(msg)
        body = dict(call.body)
        node = body
        for step in path[:-1]:
            node[step] = dict(node.get(step, {}))
            node = node[step]
        node[path[-1]] = effort
        return replace(call, body=body)

    return apply


def text_of(part: Text | Fields) -> str:
    """A text part's text, or a fields part as sorted JSON."""
    return part.text if isinstance(part, Text) else part.as_text()


def joined_text(context: Context) -> str:
    """The text parts of a context, in order; images are the caller's to carry or refuse."""
    return '\n\n'.join(text_of(p) for p in context if not isinstance(p, Image))


def is_closed(schema: Mapping) -> bool:
    """Whether a schema forbids extra properties -- the one shape OpenAI's strict mode accepts."""
    return schema.get('additionalProperties') is False
