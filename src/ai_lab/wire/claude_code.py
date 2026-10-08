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
from ai_lab.wire import text_of


def respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> CliCall:
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


def parse_response(raw: CliResult) -> dict:
    for line in reversed(raw.text.splitlines()):
        if '"type":"result"' not in line.replace(' ', ''):
            continue
        event = json.loads(line)
        if event.get('is_error') or 'structured_output' not in event:
            raise ProviderError(f'claude returned no structured output: {str(event.get("result"))[:300]}')
        return event['structured_output']
    raise ProviderError(f'claude emitted no result event; tail: {raw.text[-300:]}')
