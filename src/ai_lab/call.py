"""``python -m ai_lab.call`` -- one model call as a JSON request on stdin and a JSON reply on stdout.

The protocol is a versioned contract (:data:`CALL_SCHEMA_VERSION`; README "One call from any language"),
so a process in another language -- a Node host harness -- reaches every provider through ai-lab
without a vendor SDK, a key or a retry loop of its own. The request::

    {"verb": "respond" | "decide", "provider": str, "model": str, "task"?: str, "effort"?: str,
     "instructions": str, "context": [{"kind": "text", "text": str} | {"kind": "fields", "values": {}}],
     "schema"?: {JSON-schema object}      -- respond: required; decide: refused
     "questions"?: [question]             -- decide: required (ai_lab.spec's JSON form); respond: refused
     "images"?: [{"mime": "image/png", "data": base64}]}

and the reply, always one JSON object::

    {"schema_version": 2, "ok": bool, "kind": "" | "exhausted" | "auth" | "rate_limit" | "unsupported"
     | "timeout", "result": object | null, "usage": {...} | null, "key_slot": int, "error": str}

``result`` is the schema object (respond) or ``{"basis", "answers": {name: answer}}`` (decide). A
provider failure is CLASSIFIED, not crashed: ``ok`` is false, ``kind`` is :attr:`ProviderError.kind`
(``""`` for an unclassified one) or ``unsupported`` for :class:`Unsupported`, ``error`` names it --
never a key -- and the exit status is 0. Only a MALFORMED request (not JSON, an unknown or missing
field, a wrong type, an undeclared task, an EMPTY CONTEXT) exits non-zero (2), with the reason on stderr
and nothing on stdout -- no vendor is called. A context is empty when no part has non-blank text,
non-empty fields or an image. For ``decide`` the instructions are the first text of the context.

``--ledger PATH`` sees the client through :class:`ai_lab.ledger.Ledger` in ``record`` mode: every call is
recorded with its usage and key slot, and a request already in the file is served from it.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Final

from ai_lab.client import connect
from ai_lab.errors import ERROR_KINDS, ProviderError, Unsupported
from ai_lab.ledger import Ledger, Mode
from ai_lab.spec import (
    DecisionRequest,
    EmptyContext,
    Fields,
    Image,
    ResponseRequest,
    Text,
    decision_request_from_json,
    has_content,
)

__all__ = ['CALL_SCHEMA_VERSION', 'REPLY_KINDS', 'Malformed', 'answer']

#: The version of the reply :func:`answer` writes and of the request it reads.
CALL_SCHEMA_VERSION: Final = 2

#: Every ``kind`` a reply may carry; ``''`` is success or an unclassified provider failure.
REPLY_KINDS: Final = ERROR_KINDS | {'', 'unsupported'}

_VERBS: Final = frozenset({'respond', 'decide'})
_REQUIRED: Final = frozenset({'verb', 'provider', 'model', 'instructions', 'context'})
_OPTIONAL: Final = frozenset({'task', 'effort', 'schema', 'questions', 'images'})
_EXIT_MALFORMED: Final = 2


class Malformed(ValueError):
    """The request is not one this protocol version reads."""


def _field(request: Mapping, name: str, kind: type, default: object = None) -> object:
    value = request.get(name, default)
    if not isinstance(value, kind):
        msg = f'{name!r} must be {kind.__name__}, got {type(value).__name__}'
        raise Malformed(msg)
    return value


def _part(item: object) -> Text | Fields:
    if not isinstance(item, dict) or item.get('kind') not in {'text', 'fields'}:
        msg = f'a context item is {{"kind": "text", "text": ...}} or {{"kind": "fields", "values": ...}}, got {item!r}'
        raise Malformed(msg)
    if item['kind'] == 'text':
        return Text(_field(item, 'text', str))
    return Fields(_field(item, 'values', dict))


def _image(item: object) -> Image:
    if not isinstance(item, dict) or not isinstance(item.get('data'), str):
        msg = f'an image is {{"mime": "image/...", "data": base64}}, got {type(item).__name__}'
        raise Malformed(msg)
    try:
        return Image(base64.b64decode(item['data'], validate=True), item.get('mime', 'image/png'))
    except binascii.Error as bad:
        msg = f"an image's data is not base64: {bad}"
        raise Malformed(msg) from None


def _request(request: Mapping) -> tuple[str, DecisionRequest | ResponseRequest]:
    """The verb and the typed request; :class:`Malformed` naming the first defect."""
    if missing := sorted(_REQUIRED - set(request)):
        msg = f'missing field(s) {missing}'
        raise Malformed(msg)
    if unknown := sorted(set(request) - _REQUIRED - _OPTIONAL):
        msg = (
            f'unknown field(s) {unknown}; protocol version {CALL_SCHEMA_VERSION} reads {sorted(_REQUIRED | _OPTIONAL)}'
        )
        raise Malformed(msg)
    verb = _field(request, 'verb', str)
    if verb not in _VERBS:
        msg = f'verb {verb!r} is not one of {sorted(_VERBS)}'
        raise Malformed(msg)
    for name in ('provider', 'model', 'instructions'):
        _field(request, name, str)
    for name in ('task', 'effort'):
        if name in request:
            _field(request, name, str)
    context = tuple(_part(p) for p in _field(request, 'context', list))
    context += tuple(_image(i) for i in _field(request, 'images', list, []))
    return verb, _typed(verb, request, context)


def _typed(verb: str, request: Mapping, context: tuple) -> DecisionRequest | ResponseRequest:
    try:
        if verb == 'respond':
            if 'questions' in request or 'schema' not in request:
                msg = 'respond takes a "schema" and no "questions"'
                raise Malformed(msg)
            return ResponseRequest(request['instructions'], context, _field(request, 'schema', dict))
        if 'schema' in request or 'questions' not in request:
            msg = 'decide takes "questions" and no "schema"'
            raise Malformed(msg)
        if request['instructions']:
            context = (Text(request['instructions']), *context)
        if not has_content(context):
            msg = 'empty context: a decide needs non-blank instructions or context, or an image'
            raise EmptyContext(msg)
        questions = _field(request, 'questions', list)
        parsed = decision_request_from_json({'context': [], 'questions': questions})
        return DecisionRequest(context, parsed.questions)
    except (KeyError, TypeError, ValueError) as bad:
        if isinstance(bad, Malformed):
            raise
        if isinstance(bad, EmptyContext):
            raise Malformed(str(bad)) from None
        msg = f'the request does not describe a valid {verb}: {bad!r}'
        raise Malformed(msg) from None


def _reply(
    *, ok: bool, kind: str = '', result: object = None, usage: object = None, key_slot: int = 0, error: str = ''
) -> dict:
    return {
        'schema_version': CALL_SCHEMA_VERSION,
        'ok': ok,
        'kind': kind,
        'result': result,
        'usage': None if usage is None else asdict(usage),
        'key_slot': key_slot,
        'error': error,
    }


def answer(request: Mapping, *, ledger: str | None = None, **options: object) -> dict:
    """The reply to one *request*; :class:`Malformed` when it is not a request of this protocol."""
    if not isinstance(request, Mapping):
        msg = f'a request is a JSON object, got {type(request).__name__}'
        raise Malformed(msg)
    verb, typed = _request(request)
    spec = f'{request["provider"]}:{request["model"]}' + (f'@{request["effort"]}' if request.get('effort') else '')
    try:
        client = connect(spec, task=request.get('task'), **options)
    except Unsupported as missing:
        return _reply(ok=False, kind='unsupported', error=str(missing))
    except ValueError as bad:  # an undeclared task: the request named something no key can be under
        raise Malformed(str(bad)) from None
    if ledger is not None:
        client = Ledger(ledger, Mode.RECORD).wrap(client)
    try:
        if verb == 'respond':
            got = client.respond(typed)
            return _reply(ok=True, result=got.value, usage=got.usage, key_slot=got.key_slot)
        got = client.decide(typed)
        answers = {name: {'kind': type(a).__name__, **asdict(a)} for name, a in got.answers.items()}
        return _reply(
            ok=True, result={'basis': got.basis.value, 'answers': answers}, usage=got.usage, key_slot=got.key_slot
        )
    except Unsupported as missing:
        return _reply(ok=False, kind='unsupported', error=str(missing))
    except ProviderError as failed:
        return _reply(ok=False, kind=failed.kind, error=str(failed))


def main(argv: Sequence[str] | None = None) -> int:
    """Read one request from stdin, write one reply to stdout; exit 2 on a malformed request."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.call', description=__doc__.split('\n', 1)[0])
    parser.add_argument('--ledger', metavar='PATH', help='record every call (and serve a known one) in this ledger')
    args = parser.parse_args(argv)
    try:
        reply = answer(json.loads(sys.stdin.read()), ledger=args.ledger)
    except (json.JSONDecodeError, Malformed) as bad:
        sys.stderr.write(f'malformed request: {bad}\n')
        return _EXIT_MALFORMED
    sys.stdout.write(json.dumps(reply, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
