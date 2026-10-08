"""Every wire this package ships, in one table keyed by name.

A wire is a PROTOCOL, never a company: a vendor is a row in ``providers.toml`` that names one of
these. What each wire can do is its :class:`~ai_lab.wire.base.Wire` record -- this table holds the
records, so a provider naming anything else is refused against the set printed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from ai_lab.wire import (
    anthropic_messages,
    claude_code,
    codex,
    gemini,
    openai_chat,
    openai_chat_json,
    openai_decisions,
    openai_responses,
    typesafe,
)
from ai_lab.wire.base import Wire

__all__ = ['WIRES']

WIRES: Mapping[str, Wire] = MappingProxyType(
    {
        module.WIRE.name: module.WIRE
        for module in (
            anthropic_messages,
            claude_code,
            codex,
            gemini,
            openai_chat,
            openai_chat_json,
            openai_decisions,
            openai_responses,
            typesafe,
        )
    }
)
