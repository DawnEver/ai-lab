"""``connect('provider:model[@effort]')`` -> a :class:`Client` that decides and responds through one wire.

A client offers exactly what its wire's record carries: a native decision endpoint decides natively,
a schema-capable wire decides by emulation and responds, and anything else is :class:`Unsupported`
naming the remedy. The key is read from the environment at call time and is never stored.

EFFORT IS PART OF THE NAME. ``openai_responses:gpt-x@high`` is a different client from ``@low``: its
answers differ, so the ledger -- keyed on the name -- keeps them apart, and a comparison of efforts is
a comparison of names. A wire with no ``effort`` hook refuses one rather than ignoring it. Every
answer carries the vendor's token counts when the wire reads them (:mod:`ai_lab.usage`).

THE PROVIDER TABLE HAS TWO LAYERS AND ONE RULE. The packaged ``providers.toml`` is the baseline; a
``providers.toml`` in ``lab_commons.paths.config_root('ai_lab')`` -- ``$AI_LAB_HOME/config`` when set,
else the platform's user config directory -- adds rows and replaces a row of the same name. That is
the family's config search path, so ai-lab carries no second convention for where a user's file is.
"""

from __future__ import annotations

import os
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from lab_commons.paths import config_root

from ai_lab import emulate, transport
from ai_lab.answers import Basis, Decision, Response
from ai_lab.errors import Unsupported
from ai_lab.spec import DecisionRequest, ResponseRequest
from ai_lab.usage import Usage
from ai_lab.wire import WIRES
from ai_lab.wire.base import Wire

__all__ = ['APP', 'Client', 'Provider', 'connect', 'providers', 'user_providers_file']

#: The application name the family's path helpers resolve this package's roots under.
APP = 'ai_lab'


@dataclass(frozen=True, slots=True)
class Provider:
    """One vendor: which wire it speaks, where it is reached, which variable holds its key."""

    name: str
    wire: str
    endpoint: str
    key_env: str = ''
    vision: bool = False

    def __post_init__(self) -> None:
        """Refuse a row naming a wire this package does not ship."""
        if self.wire not in WIRES:
            msg = f'provider {self.name!r} names unknown wire {self.wire!r}; the wires are {sorted(WIRES)}'
            raise Unsupported(msg)


def user_providers_file() -> Path:
    """Where a user's own provider rows live (it need not exist)."""
    return config_root(APP) / 'providers.toml'


def _rows(text: str, origin: str) -> dict[str, Provider]:
    table = tomllib.loads(text)
    try:
        return {name: Provider(name=name, **row) for name, row in table.items()}
    except TypeError as error:
        msg = f'a provider row in {origin} has an unknown or missing field: {error}'
        raise Unsupported(msg) from None


def providers() -> dict[str, Provider]:
    """The packaged table, with every row of :func:`user_providers_file` replacing its namesake."""
    table = _rows(files('ai_lab').joinpath('providers.toml').read_text('utf-8'), 'the packaged providers.toml')
    user = user_providers_file()
    if user.is_file():
        table.update(_rows(user.read_text('utf-8'), str(user)))
    return table


class Client:
    """One provider and one model. ``post``/``run`` are injectable so a test needs no network."""

    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        effort: str = '',
        post: Callable[..., dict] = transport.post,
        run: Callable[..., transport.CliResult] = transport.run,
    ) -> None:
        """Bind ``provider`` and ``model``; refuse an HTTP provider with no model."""
        self.wire: Wire = WIRES[provider.wire]
        if not model and not self.wire.cli:
            msg = f'{provider.name} needs a model: connect("{provider.name}:<model>")'
            raise Unsupported(msg)
        if effort and self.wire.effort is None:
            msg = (
                f'{provider.name} takes no effort (wire {provider.wire}); the wires that do are '
                f'{sorted(n for n, w in WIRES.items() if w.effort)}'
            )
            raise Unsupported(msg)
        self.provider = provider
        self.model = model
        self.effort = effort
        self._post = post
        self._run = run

    @property
    def name(self) -> str:
        """``provider:model[@effort]`` -- the spelling :func:`connect` takes back."""
        return f'{self.provider.name}:{self.model}' + (f'@{self.effort}' if self.effort else '')

    def __repr__(self) -> str:
        """``Client('provider:model')`` -- never the key."""
        return f'Client({self.name!r})'

    def decide(self, request: DecisionRequest) -> Decision:
        """Answer every question: natively when the wire can, else through one derived schema."""
        self._require_vision(has_image=request.has_image)
        start = time.perf_counter()
        if self.wire.decide is not None:
            raw = self._send(self.wire.decide(request, self.model, self.provider.endpoint, self._key()))
            answers, basis, usage = self.wire.parse_decision(raw, request), Basis.NATIVE, self._usage(raw)
        else:
            value, usage = self._respond_value(emulate.as_response_request(request))
            answers, basis = emulate.parse_answers(value, request.questions), Basis.STATED
        return Decision(answers, basis, self.provider.name, self.model, time.perf_counter() - start, usage)

    def respond(self, request: ResponseRequest) -> Response:
        """An object of the request's schema."""
        self._require_vision(has_image=request.has_image)
        start = time.perf_counter()
        value, usage = self._respond_value(request)
        return Response(value, self.provider.name, self.model, time.perf_counter() - start, usage)

    def _respond_value(self, request: ResponseRequest) -> tuple[Mapping, Usage | None]:
        if self.wire.respond is None:
            msg = (
                f'{self.provider.name} is a decision endpoint and cannot respond in a schema; '
                f'use a provider whose wire responds ({sorted(n for n, w in WIRES.items() if w.respond)})'
            )
            raise Unsupported(msg)
        raw = self._send(self.wire.respond(request, self.model, self.provider.endpoint, self._key()))
        return self.wire.parse_response(raw), self._usage(raw)

    def _usage(self, raw: object) -> Usage | None:
        return None if self.wire.parse_usage is None else self.wire.parse_usage(raw)

    def _require_vision(self, *, has_image: bool) -> None:
        if has_image and not self.provider.vision:
            msg = f'{self.provider.name} does not read images; drop the Image parts or pick one that does'
            raise Unsupported(msg)

    def _key(self) -> str:
        if not self.provider.key_env:
            return ''
        key = os.environ.get(self.provider.key_env, '')
        if not key:
            msg = f'{self.provider.name} needs its key in the environment variable {self.provider.key_env}'
            raise Unsupported(msg)
        return key

    def _send(self, call: transport.HttpCall | transport.CliCall) -> object:
        if self.effort:
            call = self.wire.effort(call, self.effort)
        if isinstance(call, transport.CliCall):
            return self._run(call)
        return self._post(call)


def connect(spec: str, **options: Callable) -> Client:
    """``'openai_decisions:gpt-6-luna'``, ``'openai_responses:gpt-x@high'``, ``'claude_code'``..."""
    spec, _, effort = spec.partition('@')
    name, _, model = spec.partition(':')
    table = providers()
    if name not in table:
        msg = f'unknown provider {name!r}; known: {sorted(table)} (add one in {user_providers_file()})'
        raise Unsupported(msg)
    return Client(table[name], model, effort=effort, **options)
