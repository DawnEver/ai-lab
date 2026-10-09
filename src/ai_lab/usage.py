"""What a call consumed, as the vendor reported it.

TOKENS ARE THE FACT; COST IS DERIVED (:mod:`ai_lab.prices`). Every wire reads its vendor's usage
block into one :class:`Usage` and the ledger records it with the call's time. A CLI that reports its
own dollar figure (``claude``) carries it in :attr:`Usage.cost_usd`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    'Usage',
    'anthropic_usage',
    'chat_usage',
    'claude_code_usage',
    'gemini_usage',
    'responses_usage',
]


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens one call consumed.

    ``cached_tokens`` is part of ``input_tokens`` and ``reasoning_tokens`` part of ``output_tokens``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None


def _int(block: Mapping, *path: str) -> int:
    value: object = block
    for step in path:
        if not isinstance(value, Mapping):
            return 0
        value = value.get(step)
    return int(value) if isinstance(value, int | float) else 0


def responses_usage(raw: object) -> Usage | None:
    """OpenAI Responses and Decisions: ``usage.input_tokens`` / ``output_tokens`` with their details."""
    block = raw.get('usage') if isinstance(raw, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    return Usage(
        _int(block, 'input_tokens'),
        _int(block, 'output_tokens'),
        _int(block, 'input_tokens_details', 'cached_tokens'),
        _int(block, 'output_tokens_details', 'reasoning_tokens'),
    )


def chat_usage(raw: object) -> Usage | None:
    """Chat Completions (OpenAI and its compatibles): ``prompt_tokens`` / ``completion_tokens``."""
    block = raw.get('usage') if isinstance(raw, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    cached = _int(block, 'prompt_tokens_details', 'cached_tokens') or _int(block, 'prompt_cache_hit_tokens')
    return Usage(
        _int(block, 'prompt_tokens'),
        _int(block, 'completion_tokens'),
        cached,
        _int(block, 'completion_tokens_details', 'reasoning_tokens'),
    )


def anthropic_usage(raw: object) -> Usage | None:
    """Anthropic Messages: cache reads and writes are billed input beside ``input_tokens``."""
    block = raw.get('usage') if isinstance(raw, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    cached = _int(block, 'cache_read_input_tokens')
    written = _int(block, 'cache_creation_input_tokens')
    return Usage(_int(block, 'input_tokens') + cached + written, _int(block, 'output_tokens'), cached)


def gemini_usage(raw: object) -> Usage | None:
    """Gemini: ``usageMetadata``; thoughts are output beside the candidates."""
    block = raw.get('usageMetadata') if isinstance(raw, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    thoughts = _int(block, 'thoughtsTokenCount')
    return Usage(
        _int(block, 'promptTokenCount'),
        _int(block, 'candidatesTokenCount') + thoughts,
        _int(block, 'cachedContentTokenCount'),
        thoughts,
    )


def claude_code_usage(event: Mapping) -> Usage:
    """The ``claude`` CLI's ``result`` event: Anthropic usage plus the CLI's own ``total_cost_usd``."""
    usage = anthropic_usage(event) or Usage()
    reported = event.get('total_cost_usd')
    return Usage(
        usage.input_tokens,
        usage.output_tokens,
        usage.cached_tokens,
        usage.reasoning_tokens,
        float(reported) if isinstance(reported, int | float) else None,
    )
