"""Anthropic ``POST /v1/messages``; the schema is a single forced tool, so the reply IS its input."""

from __future__ import annotations

import base64

from ai_lab.errors import ProviderError
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.wire import text_of

_VERSION = '2023-06-01'
_MAX_TOKENS = 8192


def respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
    content = [
        {
            'type': 'image',
            'source': {'type': 'base64', 'media_type': p.mime, 'data': base64.b64encode(p.data).decode('ascii')},
        }
        if isinstance(p, Image)
        else {'type': 'text', 'text': text_of(p)}
        for p in request.context
    ]
    return HttpCall(
        url=f'{endpoint}/messages',
        body={
            'model': model,
            'max_tokens': _MAX_TOKENS,
            'system': request.instructions,
            'messages': [{'role': 'user', 'content': content}],
            'tools': [
                {'name': request.name, 'description': 'Return the answer.', 'input_schema': dict(request.schema)}
            ],
            'tool_choice': {'type': 'tool', 'name': request.name},
        },
        headers={'x-api-key': key, 'anthropic-version': _VERSION},
    )


def parse_response(raw: dict) -> dict:
    for block in raw.get('content', []):
        if block.get('type') == 'tool_use':
            return block['input']
    raise ProviderError(f'the message carries no tool_use block: stop_reason={raw.get("stop_reason")}')
