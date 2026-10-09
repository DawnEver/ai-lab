"""What a call consumed, as the vendor reported it, and what that costs at the user's own prices.

TOKENS ARE THE FACT; COST IS DERIVED. Every wire reads its vendor's usage block into one
:class:`Usage`, and the ledger records it. A price is the user's data -- vendors change them and
this package must not ship a guess -- so :func:`cost` reads ``prices.toml`` from the same config
root as ``providers.toml`` and answers ``None`` for a model it has no row for. A CLI that reports
its own dollar figure (``claude``) carries it in :attr:`Usage.cost_usd`, which wins over the table.

``prices.toml`` rows are keyed by the client name (``provider:model``, effort excluded -- effort
moves the token counts, not the rate), in US dollars per million tokens::

    ["openai_decisions:gpt-6-luna"]
    input = 0.10
    cached_input = 0.01
    output = 0.40
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from lab_commons.paths import config_root

__all__ = [
    'Usage',
    'anthropic_usage',
    'chat_usage',
    'claude_code_usage',
    'cost',
    'gemini_usage',
    'prices_file',
    'responses_usage',
]

_MILLION = 1_000_000


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


def prices_file() -> Path:
    """Where the user's prices live (it need not exist)."""
    return config_root('ai_lab') / 'prices.toml'


def _prices(path: Path | None = None) -> dict[str, Mapping[str, float]]:
    path = prices_file() if path is None else path
    return tomllib.loads(path.read_text('utf-8')) if path.is_file() else {}


def cost(usage: Usage | None, client: str, *, prices: Callable[[], Mapping] = _prices) -> float | None:
    """US dollars for ``usage`` on ``client`` (``provider:model[@effort]``), or ``None`` when unpriced."""
    if usage is None:
        return None
    if usage.cost_usd is not None:
        return usage.cost_usd
    row = prices().get(client.partition('@')[0])
    if row is None:
        return None
    fresh = usage.input_tokens - usage.cached_tokens
    cached_rate = row.get('cached_input', row['input'])
    return (fresh * row['input'] + usage.cached_tokens * cached_rate + usage.output_tokens * row['output']) / _MILLION
