"""OpenAI ``POST /v1/responses`` with a JSON-schema text format."""

from __future__ import annotations

import json

from ai_lab.errors import ProviderError
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.wire.base import Wire, is_closed, text_of

__all__ = ['WIRE']


def _respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
    content = [
        {'type': 'input_image', 'image_url': p.data_url()}
        if isinstance(p, Image)
        else {'type': 'input_text', 'text': text_of(p)}
        for p in request.context
    ]
    return HttpCall(
        url=f'{endpoint}/responses',
        body={
            'model': model,
            'instructions': request.instructions,
            'input': [{'role': 'user', 'content': content}],
            'text': {
                'format': {
                    'type': 'json_schema',
                    'name': request.name,
                    'schema': dict(request.schema),
                    'strict': is_closed(request.schema),
                }
            },
        },
        headers={'Authorization': f'Bearer {key}'},
    )


def _parse_response(raw: dict) -> dict:
    for item in raw.get('output', []):
        for part in item.get('content', []) if item.get('type') == 'message' else []:
            if part.get('type') == 'refusal':
                msg = f'the model refused: {part.get("refusal")}'
                raise ProviderError(msg)
            if part.get('type') == 'output_text':
                return json.loads(part['text'])
    msg = f'the response carries no output_text: status={raw.get("status")}'
    raise ProviderError(msg)


WIRE = Wire(name='openai_responses', cli=False, respond=_respond, parse_response=_parse_response)
