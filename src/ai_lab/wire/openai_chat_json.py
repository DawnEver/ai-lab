"""``POST /chat/completions`` with ``response_format: json_object`` -- JSON guaranteed, schema NOT enforced.

For a vendor that accepts no ``json_schema`` format (DeepSeek). The schema travels in the system
prompt, so the reply is valid JSON whose SHAPE is the model's to honour: a decision built on it is
still checked answer by answer against the declared options (:mod:`ai_lab.emulate`).
"""

from __future__ import annotations

import json

from ai_lab.spec import ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.wire.base import Wire
from ai_lab.wire.openai_chat import chat_call, parse_message

__all__ = ['WIRE']


def _respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
    schema = json.dumps(dict(request.schema), sort_keys=True)
    system = f'{request.instructions}\n\nReply with one JSON object that validates against this JSON schema:\n{schema}'
    return chat_call(request, model, endpoint, key, system=system, response_format={'type': 'json_object'})


WIRE = Wire(name='openai_chat_json', cli=False, respond=_respond, parse_response=parse_message)
