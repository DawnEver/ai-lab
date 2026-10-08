"""The local ``claude`` CLI in print mode: your own login, no API key.

Every built-in tool, MCP server and settings file is switched off and it runs in an empty
temporary directory, so it can answer but cannot act. The context is one stream-json user message,
which is the one input shape that carries images; ``--json-schema`` binds the reply and the
``result`` event's ``structured_output`` is the answer.
"""

from __future__ import annotations

import base64
import json

from ai_lab.errors import ProviderError
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import CliCall, CliResult
from ai_lab.wire.base import Wire, text_of

__all__ = ['WIRE']


def _respond(request: ResponseRequest, model: str, endpoint: str, _key: str) -> CliCall:
    content = [
        {
            'type': 'image',
            'source': {'type': 'base64', 'media_type': p.mime, 'data': base64.b64encode(p.data).decode('ascii')},
        }
        if isinstance(p, Image)
        else {'type': 'text', 'text': text_of(p)}
        for p in request.context
    ]
    message = {'type': 'user', 'message': {'role': 'user', 'content': content}}
    argv = [
        endpoint,
        '-p',
        '--input-format', 'stream-json',
        '--output-format', 'stream-json',
        '--verbose',
        '--json-schema', json.dumps(dict(request.schema)),
        '--system-prompt', request.instructions,
        '--tools', '',
        '--strict-mcp-config',
        '--setting-sources', '',
        '--no-session-persistence',
    ]  # fmt: skip
    if model:
        argv += ['--model', model]
    return CliCall(argv=tuple(argv), stdin=json.dumps(message) + '\n')


def _parse_response(raw: CliResult) -> dict:
    for line in reversed(raw.text.splitlines()):
        if '"type":"result"' not in line.replace(' ', ''):
            continue
        event = json.loads(line)
        if event.get('is_error') or 'structured_output' not in event:
            msg = f'claude returned no structured output: {str(event.get("result"))[:300]}'
            raise ProviderError(msg)
        return event['structured_output']
    msg = f'claude emitted no result event; tail: {raw.text[-300:]}'
    raise ProviderError(msg)


WIRE = Wire(name='claude_code', cli=True, respond=_respond, parse_response=_parse_response)
