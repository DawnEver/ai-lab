"""What a wire IS: a record of the translations it carries, with an absent verb spelled ``None``.

A wire is a PROTOCOL, never a company: each module in this package defines exactly one :class:`Wire`
named ``WIRE``, and :data:`ai_lab.wire.WIRES` is the one table of them. The verbs a provider has are
read off that record -- ``respond`` and ``decide`` are present or ``None`` -- so capability is the
declaration itself rather than something probed for at call time. The same holds for the two
optional hooks: ``parse_usage`` reads the vendor's token counts off a raw reply, and ``effort`` puts
a reasoning effort onto a call; a wire without one does not report usage or take an effort. And for
``batches``: a wire whose vendor takes many calls as one asynchronous batch declares :class:`Batches`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from ai_lab.answers import Answer
from ai_lab.spec import Context, DecisionRequest, Fields, Image, ResponseRequest, Text
from ai_lab.transport import HttpCall
from ai_lab.usage import Usage

__all__ = ['Batches', 'Outcome', 'Wire', 'body_effort', 'is_closed', 'joined_text', 'text_of']

Call = HttpCall
#: ``(request, model, endpoint, key) -> call``: the endpoint is the row's base URL for the wire's protocol.
Build = Callable[[ResponseRequest, str, str, str], Call]
BuildDecision = Callable[[DecisionRequest, str, str, str], Call]


@dataclass(frozen=True, slots=True)
class Outcome:
    """One call's result inside a batch: the raw reply ``parse_response`` reads, or why there is none.

    ``lapsed`` is ``'expired'`` or ``'canceled'`` when the call never ran (unbilled); ``kind`` classifies an
    errored one like :class:`~ai_lab.errors.ProviderError`.
    """

    raw: object = None
    kind: str = ''
    error: str = ''
    lapsed: str = ''


@dataclass(frozen=True, slots=True)
class Batches:
    """A vendor's asynchronous batch endpoint: many built calls as one, its state, its results by custom id.

    ``create(calls, endpoint)`` is the one POST carrying every call's body, each keyed by its custom id; ``url``
    names a batch (part ``''``) or a part of it (``'results'``, ``'cancel'``); ``ended`` and ``expires_at`` read
    a batch object; ``outcomes`` reads the results document; ``headers(key)`` authorizes a read or a cancel.
    ``custom_id`` and ``batch_id`` are the patterns an id must match before it is sent or put in a URL.
    """

    create: Callable[[Mapping[str, HttpCall], str], HttpCall]
    url: Callable[[str, str, str], str]
    ended: Callable[[Mapping], bool]
    expires_at: Callable[[Mapping], str]
    outcomes: Callable[[str], dict[str, Outcome]]
    headers: Callable[[str], Mapping[str, str]]
    custom_id: str
    batch_id: str


@dataclass(frozen=True, slots=True)
class Wire:
    """One wire of one protocol. ``decide`` is ``None`` unless the vendor answers typed questions natively.

    ``protocol`` names the API family the wire speaks (``openai``, ``anthropic``...): a provider row
    reaches it at its ``<protocol>_path``, so one account serving two protocols is one row.
    """

    name: str
    protocol: str
    respond: Build | None = None
    parse_response: Callable[[object], Mapping] | None = None
    decide: BuildDecision | None = None
    parse_decision: Callable[[object, DecisionRequest], dict[str, Answer]] | None = None
    parse_usage: Callable[[object], Usage | None] | None = None
    effort: Callable[[Call, str], Call] | None = None
    batches: Batches | None = None

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
