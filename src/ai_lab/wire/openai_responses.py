"""OpenAI ``POST /v1/responses`` with a JSON-schema text format."""

from __future__ import annotations

import json

from ai_lab.errors import ProviderError
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.wire import text_of


def strict(schema: dict) -> bool:
    """OpenAI strict mode holds only for a closed schema; anything else is sent non-strict."""
    return schema.get('additionalProperties') is False


def respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
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
                    'strict': strict(request.schema),
                }
            },
        },
        headers={'Authorization': f'Bearer {key}'},
    )


def parse_response(raw: dict) -> dict:
    for item in raw.get('output', []):
        for part in item.get('content', []) if item.get('type') == 'message' else []:
            if part.get('type') == 'refusal':
                raise ProviderError(f'the model refused: {part.get("refusal")}')
            if part.get('type') == 'output_text':
                return json.loads(part['text'])
    raise ProviderError(f'the response carries no output_text: status={raw.get("status")}')
