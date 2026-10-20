"""``POST /chat/completions`` with ``response_format: json_object`` -- JSON guaranteed, schema NOT enforced.

For a vendor that accepts no ``json_schema`` format (DeepSeek). The schema travels in the system
prompt, so the reply is valid JSON whose SHAPE is the model's to honour: a decision built on it is
still checked answer by answer against the declared options (:mod:`ai_lab.emulate`).
"""

from __future__ import annotations

import json
from dataclasses import replace

from ai_lab.spec import ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.usage import chat_usage
from ai_lab.wire.base import Call, Wire, body_effort
from ai_lab.wire.openai_chat import chat_call, parse_message

__all__ = ['WIRE']


def _respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
    schema = json.dumps(dict(request.schema), sort_keys=True)
    system = f'{request.instructions}\n\nReply with one JSON object that validates against this JSON schema:\n{schema}'
    return chat_call(request, model, endpoint, key, system=system, response_format={'type': 'json_object'})


def _effort(call: Call, effort: str) -> Call:
    """``off`` turns thinking off (``thinking: disabled``); a level turns it on at that ``reasoning_effort``.

    MEASURED 2026-10-14: deepseek-v4-flash thinks by DEFAULT -- 26k output tokens per decision whose
    answer is a few hundred tokens of JSON -- so a decider that must not think has to say so.
    """
    if not isinstance(call, HttpCall):
        msg = 'a body effort applies to an HTTP call'
        raise TypeError(msg)
    if effort == 'off':
        return replace(call, body={**call.body, 'thinking': {'type': 'disabled'}})
    thinking = replace(call, body={**call.body, 'thinking': {'type': 'enabled'}})
    return body_effort('reasoning_effort')(thinking, effort)


WIRE = Wire(
    name='openai_chat_json',
    respond=_respond,
    parse_response=parse_message,
    parse_usage=chat_usage,
    effort=_effort,
)
