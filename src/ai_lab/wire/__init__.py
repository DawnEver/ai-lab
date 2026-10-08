"""One module per wire PROTOCOL, never per company: a vendor is a row in ``providers.toml``.

A wire module is pure translation and declares its verbs by defining them:

* ``respond(request, model, endpoint, key) -> HttpCall | CliCall`` and ``parse_response(raw) -> Mapping``
* ``decide(request, model, endpoint, key) -> HttpCall | CliCall`` and ``parse_decision(raw, request) -> answers``

A wire without ``decide`` still decides -- through ``respond`` and :mod:`ai_lab.emulate` -- so only
a native decision endpoint defines it. Capability is therefore read off the code, not declared
beside it.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from ai_lab.spec import Context, Fields, Image, Text

__all__ = ['WIRES', 'load', 'text_of']

#: Every wire this package ships. A provider row naming anything else is refused.
WIRES = frozenset(
    {
        'anthropic_messages',
        'claude_code',
        'codex',
        'gemini',
        'openai_chat',
        'openai_decisions',
        'openai_responses',
        'typesafe',
    }
)


def load(name: str) -> ModuleType:
    if name not in WIRES:
        raise ValueError(f'unknown wire {name!r}; the wires are {sorted(WIRES)}')
    return importlib.import_module(f'ai_lab.wire.{name}')


def text_of(part: Text | Fields) -> str:
    return part.text if isinstance(part, Text) else part.as_text()


def joined_text(context: Context) -> str:
    """The text parts of a context, in order; images are the caller's to carry or refuse."""
    return '\n\n'.join(text_of(p) for p in context if not isinstance(p, Image))
