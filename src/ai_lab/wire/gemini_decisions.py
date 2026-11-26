"""Google ``POST /v1beta/decisions`` (``gemini-decision-1``): native noul / choice / score (:mod:`ai_lab.wire.noul`).

Documented at aistudio.google.com/docs/decisions-api (read 2026-11-12): ``input`` is text, or content blocks when
an image is present; the reply is ``{answers, usage}`` and only input tokens are billed. A safety block is an HTTP
400, never an answer.
"""

from __future__ import annotations

import base64

from ai_lab.spec import DecisionRequest, Image
from ai_lab.transport import HttpCall
from ai_lab.usage import decisions_usage
from ai_lab.wire import noul
from ai_lab.wire.base import Wire, text_of

__all__ = ['WIRE']


def _input(request: DecisionRequest) -> str | list:
    if not request.has_image:
        return '\n\n'.join(text_of(p) for p in request.context)
    return [
        {'type': 'image', 'data': base64.b64encode(p.data).decode('ascii'), 'mime_type': p.mime}
        if isinstance(p, Image)
        else {'type': 'text', 'text': text_of(p)}
        for p in request.context
    ]


def _decide(request: DecisionRequest, model: str, endpoint: str, key: str) -> HttpCall:
    return HttpCall(
        url=f'{endpoint}/decisions',
        body={'model': model, 'input': _input(request), 'questions': noul.questions(request)},
        headers={'x-goog-api-key': key},
    )


WIRE = Wire(
    name='gemini_decisions',
    protocol='gemini',
    decide=_decide,
    parse_decision=noul.parse_decision,
    parse_usage=decisions_usage,
)
