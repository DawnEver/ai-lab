"""``POST /chat/completions`` with ``response_format: json_schema``.

The protocol most vendors speak (OpenRouter, Ollama, vLLM, DeepSeek, Qwen, Groq, Together...).
"""

from __future__ import annotations

import json

from ai_lab.errors import ProviderError
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.wire.base import Wire, is_closed, text_of

__all__ = ['WIRE']


def _respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
    content = [
        {'type': 'image_url', 'image_url': {'url': p.data_url()}}
        if isinstance(p, Image)
        else {'type': 'text', 'text': text_of(p)}
        for p in request.context
    ]
    return HttpCall(
        url=f'{endpoint}/chat/completions',
        body={
            'model': model,
            'messages': [
                {'role': 'system', 'content': request.instructions},
                {'role': 'user', 'content': content},
            ],
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': request.name,
                    'schema': dict(request.schema),
                    'strict': is_closed(request.schema),
                },
            },
        },
        headers={'Authorization': f'Bearer {key}'} if key else {},
    )


def _parse_response(raw: dict) -> dict:
    try:
        message = raw['choices'][0]['message']
    except (KeyError, IndexError):
        msg = f'the completion carries no message: {str(raw)[:300]}'
        raise ProviderError(msg) from None
    if message.get('refusal'):
        msg = f'the model refused: {message["refusal"]}'
        raise ProviderError(msg)
    try:
        return json.loads(message['content'])
    except (TypeError, json.JSONDecodeError):
        msg = f'the message is not the declared JSON: {str(message.get("content"))[:300]}'
        raise ProviderError(msg) from None


WIRE = Wire(name='openai_chat', cli=False, respond=_respond, parse_response=_parse_response)
