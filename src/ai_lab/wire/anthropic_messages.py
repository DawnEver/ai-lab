"""Anthropic ``POST /v1/messages``; the schema is bound as structured output, so the reply IS the object.

``output_config.format`` (``json_schema``) is the documented binding on every current model; a forced
``tool_choice`` is refused by the Claude 5.5 models ("type tool and any are not supported for this model",
measured 2026-10-16). Top-level ``cache_control`` turns on automatic prompt caching: the repeated
instructions and context prefix are read back at 0.05-0.1x the input rate once a request has written them;
a second breakpoint on the block before the newest lets an append-only context read what the previous
call wrote (the automatic one sits on the newest block, which no later request repeats).
"""

from __future__ import annotations

import base64
import json

from ai_lab.errors import ProviderError
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.usage import anthropic_usage
from ai_lab.wire.base import Wire, text_of

__all__ = ['WIRE']

_VERSION = '2023-06-01'
_MAX_TOKENS = 8192


def _respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
    content = [
        {
            'type': 'image',
            'source': {'type': 'base64', 'media_type': p.mime, 'data': base64.b64encode(p.data).decode('ascii')},
        }
        if isinstance(p, Image)
        else {'type': 'text', 'text': text_of(p)}
        for p in request.context
    ]
    if len(content) > 1:
        # an append-only context's next request repeats every block but the newest: break there
        content[-2]['cache_control'] = {'type': 'ephemeral'}
    return HttpCall(
        url=f'{endpoint}/messages',
        body={
            'model': model,
            'max_tokens': _MAX_TOKENS,
            'system': request.instructions,
            'messages': [{'role': 'user', 'content': content}],
            'output_config': {'format': {'type': 'json_schema', 'schema': dict(request.schema)}},
            'cache_control': {'type': 'ephemeral'},
        },
        headers={'x-api-key': key, 'anthropic-version': _VERSION},
    )


def _parse_response(raw: dict) -> dict:
    text = ''.join(b.get('text', '') for b in raw.get('content', []) if b.get('type') == 'text')
    if not text:
        msg = f'the message carries no text: stop_reason={raw.get("stop_reason")}'
        raise ProviderError(msg)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f'the structured output is not JSON ({exc}): stop_reason={raw.get("stop_reason")}'
        raise ProviderError(msg) from exc


WIRE = Wire(
    name='anthropic_messages',
    protocol='anthropic',
    respond=_respond,
    parse_response=_parse_response,
    parse_usage=anthropic_usage,
)
