"""``POST /chat/completions`` with ``response_format: json_schema`` -- the protocol most vendors
speak (OpenRouter, Ollama, vLLM, DeepSeek, Qwen, Groq, Together...)."""

from __future__ import annotations

import json

from ai_lab.errors import ProviderError
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.wire import text_of
from ai_lab.wire.openai_responses import strict


def respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
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
                'json_schema': {'name': request.name, 'schema': dict(request.schema), 'strict': strict(request.schema)},
            },
        },
        headers={'Authorization': f'Bearer {key}'} if key else {},
    )


def parse_response(raw: dict) -> dict:
    try:
        message = raw['choices'][0]['message']
    except (KeyError, IndexError):
        raise ProviderError(f'the completion carries no message: {str(raw)[:300]}') from None
    if message.get('refusal'):
        raise ProviderError(f'the model refused: {message["refusal"]}')
    try:
        return json.loads(message['content'])
    except (TypeError, json.JSONDecodeError):
        raise ProviderError(f'the message is not the declared JSON: {str(message.get("content"))[:300]}') from None
