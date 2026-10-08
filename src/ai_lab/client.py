"""``connect('provider:model')`` -> a :class:`Client` that decides and responds through one wire.

A client offers exactly what its wire can do: a native decision endpoint decides natively, a
schema-capable wire decides by emulation and responds, and anything else is :class:`Unsupported`
naming the remedy. The key is read from the environment at call time and is never stored.
"""

from __future__ import annotations

import os
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from ai_lab import emulate, transport, wire
from ai_lab.answers import Basis, Decision, Response
from ai_lab.errors import Unsupported
from ai_lab.spec import DecisionRequest, ResponseRequest

__all__ = ['Client', 'Provider', 'connect', 'providers']

PROVIDERS_ENV = 'AI_LAB_PROVIDERS'
_CLI_WIRES = frozenset({'claude_code', 'codex'})


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    wire: str
    endpoint: str
    key_env: str = ''
    vision: bool = False

    def __post_init__(self) -> None:
        wire.load(self.wire)

    @property
    def is_cli(self) -> bool:
        return self.wire in _CLI_WIRES


def _rows(text: str, origin: str) -> dict[str, Provider]:
    table = tomllib.loads(text)
    try:
        return {name: Provider(name=name, **row) for name, row in table.items()}
    except TypeError as error:
        raise ValueError(f'a provider row in {origin} has an unknown or missing field: {error}') from None


def providers() -> dict[str, Provider]:
    """The packaged table, with every row of ``$AI_LAB_PROVIDERS`` (if set) replacing its namesake."""
    table = _rows(files('ai_lab').joinpath('providers.toml').read_text('utf-8'), 'the packaged providers.toml')
    extra = os.environ.get(PROVIDERS_ENV)
    if extra:
        table.update(_rows(Path(extra).read_text('utf-8'), extra))
    return table


class Client:
    """One provider and one model. ``post``/``run`` are injectable so a test needs no network."""

    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        post: Callable[..., dict] = transport.post,
        run: Callable[..., transport.CliResult] = transport.run,
        timeout_s: float | None = None,
    ) -> None:
        if not model and not provider.is_cli:
            raise ValueError(f'{provider.name} needs a model: connect("{provider.name}:<model>")')
        self.provider = provider
        self.model = model
        self._wire = wire.load(provider.wire)
        self._post = post
        self._run = run
        self._timeout = {} if timeout_s is None else {'timeout_s': timeout_s}

    @property
    def name(self) -> str:
        return f'{self.provider.name}:{self.model}'

    @property
    def decides_natively(self) -> bool:
        return hasattr(self._wire, 'decide')

    @property
    def responds(self) -> bool:
        return hasattr(self._wire, 'respond')

    def __repr__(self) -> str:
        return f'Client({self.name!r})'

    def decide(self, request: DecisionRequest) -> Decision:
        self._require_vision(request.has_image)
        start = time.perf_counter()
        if self.decides_natively:
            raw = self._send(self._wire.decide(request, self.model, self.provider.endpoint, self._key()))
            answers, basis = self._wire.parse_decision(raw, request), Basis.NATIVE
        elif self.responds:
            value = self._respond_value(emulate.as_response_request(request))
            answers, basis = emulate.parse_answers(value, request.questions), Basis.STATED
        else:
            raise Unsupported(f'{self.provider.name} can neither decide nor respond')
        return Decision(answers, basis, self.provider.name, self.model, time.perf_counter() - start)

    def respond(self, request: ResponseRequest) -> Response:
        if not self.responds:
            raise Unsupported(
                f'{self.provider.name} is a decision endpoint and cannot respond in a schema; '
                'use a provider whose wire responds (e.g. openai, anthropic, claude_code)'
            )
        self._require_vision(request.has_image)
        start = time.perf_counter()
        value = self._respond_value(request)
        return Response(value, self.provider.name, self.model, time.perf_counter() - start)

    def _respond_value(self, request: ResponseRequest) -> Mapping:
        raw = self._send(self._wire.respond(request, self.model, self.provider.endpoint, self._key()))
        return self._wire.parse_response(raw)

    def _require_vision(self, has_image: bool) -> None:
        if has_image and not self.provider.vision:
            raise Unsupported(f'{self.provider.name} does not read images; drop the Image parts or pick one that does')

    def _key(self) -> str:
        if not self.provider.key_env:
            return ''
        key = os.environ.get(self.provider.key_env, '')
        if not key:
            raise Unsupported(f'{self.provider.name} needs its key in the environment variable {self.provider.key_env}')
        return key

    def _send(self, call: transport.HttpCall | transport.CliCall) -> object:
        if isinstance(call, transport.CliCall):
            return self._run(call, **self._timeout)
        return self._post(call, **self._timeout)


def connect(spec: str, **options: object) -> Client:
    """``'openai_decisions:gpt-6-luna'``, ``'anthropic:claude-sonnet-5-5'``, ``'claude_code'``..."""
    name, _, model = spec.partition(':')
    table = providers()
    if name not in table:
        raise Unsupported(f'unknown provider {name!r}; known: {sorted(table)} (add one via ${PROVIDERS_ENV})')
    return Client(table[name], model, **options)
