"""Google ``models/{model}:generateContent`` with ``responseJsonSchema``."""

from __future__ import annotations

import base64
import json

from ai_lab.errors import ProviderError
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.wire.base import Wire, text_of

__all__ = ['WIRE']


def _respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
    parts = [
        {'inlineData': {'mimeType': p.mime, 'data': base64.b64encode(p.data).decode('ascii')}}
        if isinstance(p, Image)
        else {'text': text_of(p)}
        for p in request.context
    ]
    return HttpCall(
        url=f'{endpoint}/models/{model}:generateContent',
        body={
            'systemInstruction': {'parts': [{'text': request.instructions}]},
            'contents': [{'role': 'user', 'parts': parts}],
            'generationConfig': {'responseMimeType': 'application/json', 'responseJsonSchema': dict(request.schema)},
        },
        headers={'x-goog-api-key': key},
    )


def _parse_response(raw: dict) -> dict:
    try:
        text = ''.join(p.get('text', '') for p in raw['candidates'][0]['content']['parts'])
        return json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError):
        reason = (raw.get('candidates') or [{}])[0].get('finishReason') or raw.get('promptFeedback')
        msg = f'the candidate is not the declared JSON (finish: {reason})'
        raise ProviderError(msg) from None


WIRE = Wire(name='gemini', cli=False, respond=_respond, parse_response=_parse_response)
