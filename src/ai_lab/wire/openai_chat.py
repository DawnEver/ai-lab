"""``POST /chat/completions`` with ``response_format: json_schema`` -- the schema is ENFORCED.

The protocol most vendors speak (OpenRouter, Ollama, vLLM, Qwen, Groq, Together...). A vendor that
accepts only ``json_object`` speaks :mod:`ai_lab.wire.openai_chat_json` instead.
"""

from __future__ import annotations

import json

from ai_lab.errors import ProviderError
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.usage import chat_usage
from ai_lab.wire.base import Wire, body_effort, is_closed, text_of

__all__ = ['WIRE', 'chat_call', 'parse_message']


def chat_call(
    request: ResponseRequest, model: str, endpoint: str, key: str, *, system: str, response_format: dict
) -> HttpCall:
    """One chat completion: ``system`` first, the context as one user message, then ``response_format``."""
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
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
            'response_format': response_format,
        },
        headers={'Authorization': f'Bearer {key}'} if key else {},
    )


def _respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
    schema = {'name': request.name, 'schema': dict(request.schema), 'strict': is_closed(request.schema)}
    return chat_call(
        request,
        model,
        endpoint,
        key,
        system=request.instructions,
        response_format={'type': 'json_schema', 'json_schema': schema},
    )


def parse_message(raw: dict) -> dict:
    """The first choice's content as JSON; a refusal or non-JSON content is a ProviderError."""
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


WIRE = Wire(
    name='openai_chat',
    cli=False,
    respond=_respond,
    parse_response=parse_message,
    parse_usage=chat_usage,
    effort=body_effort('reasoning_effort'),
)
