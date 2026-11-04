"""Anthropic ``POST /v1/messages``; the schema is bound as structured output, so the reply IS the object.

``output_config.format`` (``json_schema``) is the documented binding on every current model; a forced
``tool_choice`` is refused by the Claude 5.5 models ("type tool and any are not supported for this model",
measured 2026-10-16). Top-level ``cache_control`` turns on automatic prompt caching: the repeated
instructions and context prefix are read back at 0.05-0.1x the input rate once a request has written them;
a breakpoint on the FIRST block caches the stable prefix a caller puts there (a review's rules, map and diff,
which every reviewer, turn and slice of one change shares), and one on the block before the newest lets an
append-only context read what the previous call wrote (the automatic one sits on the newest block, which no
later request repeats): three of the four breakpoints a request may carry.
A reply cut at its output budget (``stop_reason=max_tokens``) is a :class:`ProviderError`, never a parsed prefix.
Anthropic refuses an object schema whose ``additionalProperties`` is not explicitly ``false`` (HTTP 400,
measured 2026-10-21), so the wire sends a closed copy of the caller's schema.

MESSAGE BATCHES (:data:`BATCHES`, https://platform.claude.com/docs/en/build-with-claude/batch-processing): the
same bodies, each the ``params`` of one ``{custom_id, params}`` request, POSTed together to ``/messages/batches``
at half the price; the batch reads ``processing_status`` ``ended`` when every request has a result (most within an
hour, expired at 24 h), and ``/results`` is JSONL in ANY order, matched by ``custom_id``. A result is
``succeeded`` (the message), ``errored`` (unbilled; the error's type classifies it), ``canceled`` or ``expired``
(unbilled, never sent to the model). The bodies keep their ``cache_control``: caching stacks with the batch
discount, best-effort across a batch's concurrent requests.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping

from ai_lab.errors import ProviderError, Unsupported
from ai_lab.spec import Image, ResponseRequest
from ai_lab.transport import HttpCall
from ai_lab.usage import anthropic_usage
from ai_lab.wire.base import Batches, Outcome, Wire, text_of

__all__ = ['BATCHES', 'WIRE']

_VERSION = '2023-06-01'
_MAX_TOKENS = 8192


def _headers(key: str) -> dict[str, str]:
    return {'x-api-key': key, 'anthropic-version': _VERSION}


def _closed(node: object) -> object:
    """A copy of *node* with ``additionalProperties: false`` on every object node, at any depth.

    An object that asks for anything else (an open map) is one Anthropic cannot bind: :class:`Unsupported`.
    """
    if isinstance(node, list):
        return [_closed(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {k: _closed(v) for k, v in node.items()}
    kind = out.get('type')
    if kind == 'object' or (isinstance(kind, list) and 'object' in kind):
        if out.get('additionalProperties', False) is not False:
            msg = (
                f'Anthropic structured output binds only closed objects; this one sets {out["additionalProperties"]!r}'
            )
            raise Unsupported(msg)
        out['additionalProperties'] = False
    return out


def _respond(request: ResponseRequest, model: str, endpoint: str, key: str) -> HttpCall:
    content = [
        {
            'type': 'image',
            'source': {'type': 'base64', 'media_type': p.mime, 'data': base64.b64encode(p.data).decode('ascii')},
        }
        if isinstance(p, Image)
        else {'type': 'text', 'text': text_of(p)}
        for p in request.context
    ]
    for at in sorted({0, len(content) - 2}):
        if 0 <= at < len(content) - 1:
            # the stable prefix, and the block an append-only context's next request repeats last
            content[at]['cache_control'] = {'type': 'ephemeral'}
    return HttpCall(
        url=f'{endpoint}/messages',
        body={
            'model': model,
            'max_tokens': _MAX_TOKENS,
            'system': request.instructions,
            'messages': [{'role': 'user', 'content': content}],
            'output_config': {'format': {'type': 'json_schema', 'schema': _closed(request.schema)}},
            'cache_control': {'type': 'ephemeral'},
        },
        headers=_headers(key),
    )


def _parse_response(raw: dict) -> dict:
    if raw.get('stop_reason') == 'max_tokens':
        msg = 'the reply was cut at its output budget: stop_reason=max_tokens; raise the request max_tokens'
        raise ProviderError(msg)
    text = ''.join(b.get('text', '') for b in raw.get('content', []) if b.get('type') == 'text')
    if not text:
        msg = f'the message carries no text: stop_reason={raw.get("stop_reason")}'
        raise ProviderError(msg)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f'the structured output is not JSON ({exc}): stop_reason={raw.get("stop_reason")}'
        raise ProviderError(msg) from exc


#: An errored batch result's error type, as the kind the same failure carries on a synchronous call.
_ERROR_KINDS = {
    'authentication_error': 'auth', 'permission_error': 'auth', 'billing_error': 'exhausted',
    'rate_limit_error': 'rate_limit', 'timeout_error': 'timeout',
}  # fmt: skip


def _batch_create(calls: Mapping[str, HttpCall], endpoint: str) -> HttpCall:
    headers = next(iter(calls.values())).headers if calls else {}
    requests = [{'custom_id': cid, 'params': dict(call.body)} for cid, call in calls.items()]
    return HttpCall(url=f'{endpoint}/messages/batches', body={'requests': requests}, headers=headers)


def _batch_url(endpoint: str, batch_id: str, part: str) -> str:
    return f'{endpoint}/messages/batches/{batch_id}' + (f'/{part}' if part else '')


def _error_of(result: Mapping) -> tuple[str, str]:
    """The innermost ``{type, message}`` of an errored result (``error.error`` in the API's envelope)."""
    error = result.get('error')
    while isinstance(error, Mapping) and isinstance(error.get('error'), Mapping):
        error = error['error']
    if not isinstance(error, Mapping):
        return '', 'errored with no error object'
    return str(error.get('type', '')), str(error.get('message', ''))


def _outcomes(text: str) -> dict[str, Outcome]:
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result = row.get('result') or {}
        kind = result.get('type')
        if kind == 'succeeded':
            out[row['custom_id']] = Outcome(raw=result.get('message'))
        elif kind in {'expired', 'canceled'}:
            out[row['custom_id']] = Outcome(lapsed=kind, error=f'the batch request {kind} before it ran')
        else:
            error_type, message = _error_of(result)
            error = f'batch request errored: {error_type}: {message}'[:500]
            out[row['custom_id']] = Outcome(kind=_ERROR_KINDS.get(error_type, ''), error=error)
    return out


BATCHES = Batches(
    create=_batch_create,
    url=_batch_url,
    ended=lambda batch: batch.get('processing_status') == 'ended',
    expires_at=lambda batch: str(batch.get('expires_at') or ''),
    outcomes=_outcomes,
    headers=_headers,
    custom_id=r'[a-zA-Z0-9_-]{1,64}',
    batch_id=r'[A-Za-z0-9_-]{1,128}',
)


WIRE = Wire(
    name='anthropic_messages',
    protocol='anthropic',
    respond=_respond,
    parse_response=_parse_response,
    parse_usage=anthropic_usage,
    batches=BATCHES,
)
