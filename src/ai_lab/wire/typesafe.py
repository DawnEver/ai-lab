"""TypeSafe ``POST /v1/systemone`` (Jev): native noul / choice / score with probabilities (:mod:`ai_lab.wire.noul`).

The state is sent as an OBJECT when the context is a single :class:`Fields`, else as text.
"""

from __future__ import annotations

from ai_lab.spec import DecisionRequest, Fields
from ai_lab.transport import HttpCall
from ai_lab.usage import responses_usage
from ai_lab.wire import noul
from ai_lab.wire.base import Wire, joined_text

__all__ = ['WIRE']


def _decide(request: DecisionRequest, model: str, endpoint: str, key: str) -> HttpCall:
    context = request.context
    state = dict(context[0].values) if len(context) == 1 and isinstance(context[0], Fields) else joined_text(context)
    return HttpCall(
        url=f'{endpoint}/systemone',
        body={'model': model, 'state': state, 'questions': noul.questions(request)},
        headers={'Authorization': f'Bearer {key}'},
    )


#: Usage measured 2026-10-09: `usage.input_tokens` / `output_tokens`, the Responses shape.
WIRE = Wire(
    name='typesafe',
    protocol='typesafe',
    decide=_decide,
    parse_decision=noul.parse_decision,
    parse_usage=responses_usage,
)
