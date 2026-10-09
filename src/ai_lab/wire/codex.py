"""The local ``codex exec``: your own login, no API key.

Read-only sandbox, ephemeral session, an empty temporary working directory; ``--output-schema``
binds the final message and ``-o`` writes it to a file this wire reads back. Images travel as files.
"""

from __future__ import annotations

import json
import mimetypes
from dataclasses import replace

from ai_lab.errors import ProviderError
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import CliCall, CliResult
from ai_lab.wire.base import Wire, text_of

__all__ = ['WIRE']

_OUT = 'answer.json'
_SCHEMA = 'schema.json'


def _respond(request: ResponseRequest, model: str, endpoint: str, _key: str) -> CliCall:
    files = {_SCHEMA: json.dumps(dict(request.schema)).encode('utf-8')}
    images: list[str] = []
    texts = [request.instructions]
    for index, part in enumerate(request.context):
        if isinstance(part, Image):
            name = f'image{index}{mimetypes.guess_extension(part.mime) or ".png"}'
            files[name] = part.data
            images += ['-i', f'{{dir}}/{name}']
        else:
            texts.append(text_of(part))
    argv = [
        endpoint, 'exec',
        '--sandbox', 'read-only',
        '--skip-git-repo-check',
        '--ephemeral',
        '-C', '{dir}',
        '--output-schema', f'{{dir}}/{_SCHEMA}',
        '-o', f'{{dir}}/{_OUT}',
        *images,
    ]  # fmt: skip
    if model:
        argv += ['-m', model]
    argv.append('-')
    return CliCall(argv=tuple(argv), stdin='\n\n'.join(texts), files=files, output=_OUT)


def _parse_response(raw: CliResult) -> dict:
    try:
        return json.loads(raw.text)
    except json.JSONDecodeError:
        msg = f'codex wrote a final message that is not JSON: {raw.text[:300]}'
        raise ProviderError(msg) from None


def _effort(call: CliCall, effort: str) -> CliCall:
    """``-c model_reasoning_effort=<effort>``, before the trailing ``-`` that reads the prompt."""
    return replace(call, argv=(*call.argv[:-1], '-c', f'model_reasoning_effort="{effort}"', call.argv[-1]))


#: No ``parse_usage``: the answer is read from the ``-o`` file, which carries no token counts.
WIRE = Wire(name='codex', cli=True, respond=_respond, parse_response=_parse_response, effort=_effort)
