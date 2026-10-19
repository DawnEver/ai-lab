"""``connect('provider:model[@effort]')`` -> a :class:`Client` that decides and responds through one wire.

A client offers exactly what its wire's record carries: a native decision endpoint decides natively,
a schema-capable wire decides by emulation and responds, and anything else is :class:`Unsupported`
naming the remedy. The key is read at call time from the git credential store, keyed by the host of
the provider's endpoint, and is never held, logged or put in a repr (:mod:`ai_lab.auth` stores it).

EFFORT IS PART OF THE NAME. ``openai_responses:gpt-x@high`` is a different client from ``@low``: its
answers differ, so the ledger -- keyed on the name -- keeps them apart, and a comparison of efforts is
a comparison of names. A wire with no ``effort`` hook refuses one rather than ignoring it. Every
answer carries the vendor's token counts when the wire reads them (:mod:`ai_lab.usage`).

THE PROVIDER TABLE HAS TWO LAYERS AND ONE RULE. The packaged ``providers.toml`` is the baseline; a
``providers.toml`` in ``lab_commons.paths.config_root('ai_lab')`` -- ``$AI_LAB_HOME/config`` when set,
else the platform's user config directory -- adds rows and replaces a row of the same name. That is
the family's config search path, so ai-lab carries no second convention for where a user's file is.

EVERY ROW DECLARES ITS AUTH KIND (:data:`AUTH_KINDS`); a missing or unknown one is refused, never
inferred from an empty ``key_name``. A ``subscription`` row (the logged-in ``claude`` / ``codex`` CLIs)
is LOCAL ONLY: a request through one on a Slurm compute node (``SLURM_JOB_ID`` set) is refused, naming
the row's ``api_provider`` -- the same vendor's keyed ``api`` row -- as the one to use there.

ONE PROVIDER IS ONE ACCOUNT KIND OF ONE VENDOR (user ruling 2026-10-16). A vendor endpoint that only
some models of the account serve (OpenAI's ``/decisions``) is not a second row: the row declares
``model_wires = {model = wire}`` and :meth:`Provider.wire_for` picks the wire by the MODEL, as data.
A retired provider spelling (:data:`RETIRED_PROVIDERS`) is refused naming its replacement.

AN API ACCOUNT'S KEYS ARE ORDERED SLOTS: usernames ``api-key``, ``api-key-2``, ``api-key-3``... under
the endpoint host, read up to the first missing slot. A call starts on slot 1 and moves to the next
slot when it fails at the ACCOUNT level (``ProviderError.kind``: exhausted, auth, rate limit); when
every slot fails, the error lists each slot's failure kind -- never a key. The slot that served a
call is on its :class:`~ai_lab.answers.Decision` / :class:`~ai_lab.answers.Response`.

A TASK OWNS ITS KEY (user ruling 2026-10-19). ``connect(spec, task='optimization')`` sends only that
task's key -- username ``api-key-<task>`` under the same host, a declared task of
:data:`lab_commons.credentials.API_KEY_TASKS` -- with no slots and never the default key: missing, the
call is refused naming ``login <provider> --task <task>``. Its answers report slot 1, its one key.
"""

from __future__ import annotations

import os
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from lab_commons.credentials import API_KEY_USERNAME, NoCredential, api_key, credential_for, task_username
from lab_commons.paths import config_root

from ai_lab import emulate, transport
from ai_lab.answers import Basis, Decision, Response
from ai_lab.errors import ProviderError, Unsupported
from ai_lab.spec import DecisionRequest, ResponseRequest
from ai_lab.usage import Usage
from ai_lab.wire import WIRES
from ai_lab.wire.base import Wire

__all__ = [
    'APP',
    'AUTH_KINDS',
    'RETIRED_PROVIDERS',
    'Client',
    'Provider',
    'connect',
    'providers',
    'slot_username',
    'stored_slots',
    'user_providers_file',
]

#: The application name the family's path helpers resolve this package's roots under.
APP = 'ai_lab'


#: How a provider authenticates: a keyed per-token ``api`` account, a logged-in vendor CLI on a
#: ``subscription`` (local only), or a keyless ``local`` server you run yourself.
AUTH_KINDS = frozenset({'api', 'subscription', 'local'})


@dataclass(frozen=True, slots=True)
class Provider:
    """One vendor: which wire it speaks, where it is reached, and whether it needs a key.

    ``key_name`` is the conventional name of the vendor's key (``OPENAI_API_KEY``); ``''`` is keyless.
    It is never looked up in the environment: the key itself lives in the git credential store under
    :attr:`host`, and the name only lets ``python -m ai_lab.auth login --from-file`` import a key file.
    """

    name: str
    wire: str
    endpoint: str
    #: One of :data:`AUTH_KINDS`.
    auth: str
    key_name: str = ''
    vision: bool = False
    #: Who bills this provider's tokens, as the price catalogs name it; ``''`` is unmetered.
    vendor: str = ''
    #: For a ``subscription`` row, the same vendor's ``api`` row -- the alternative off the laptop.
    api_provider: str = ''
    #: ``{model: wire}`` -- a model of this account served by another wire than :attr:`wire`.
    model_wires: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a row naming a wire this package does not ship, or an auth kind it does not know."""
        for wire in (self.wire, *self.model_wires.values()):
            if wire not in WIRES:
                msg = f'provider {self.name!r} names unknown wire {wire!r}; the wires are {sorted(WIRES)}'
                raise Unsupported(msg)
        if self.auth not in AUTH_KINDS:
            msg = f'provider {self.name!r} names unknown auth {self.auth!r}; the kinds are {sorted(AUTH_KINDS)}'
            raise Unsupported(msg)
        if (self.auth == 'subscription') != bool(self.api_provider):
            msg = f'provider {self.name!r}: api_provider is declared by, and only by, a subscription row'
            raise Unsupported(msg)

    def wire_for(self, model: str) -> str:
        """The wire that serves *model*: its declared :attr:`model_wires` entry, else :attr:`wire`."""
        return self.model_wires.get(model, self.wire)

    @property
    def keyed(self) -> bool:
        """Whether a call needs a stored key."""
        return bool(self.key_name)

    @property
    def host(self) -> str:
        """The host the key is stored under: providers sharing it share one key."""
        return urlsplit(self.endpoint).hostname or ''


#: The field a key USED to be named by (an environment variable), retired 2026-10-13 for the store.
_RETIRED_KEY_FIELD = 'key_env'

#: Provider names that USED to exist, each with the provider that replaced it -- refused, never aliased.
#: ``openai_decisions`` was a second door into the one OpenAI API account (retired 2026-10-16): its
#: wire is now chosen by ``openai``'s ``model_wires``.
RETIRED_PROVIDERS: Mapping[str, str] = {'openai_decisions': 'openai'}


def _refuse_retired(name: str, model: str, origin: str) -> None:
    if name in RETIRED_PROVIDERS:
        replacement = f'{RETIRED_PROVIDERS[name]}:{model}' if model else RETIRED_PROVIDERS[name]
        msg = f'provider {name!r} ({origin}) is retired: one provider is one account; use {replacement!r}'
        raise Unsupported(msg)


def slot_username(slot: int) -> str:
    """The credential-store username of key slot *slot*: ``api-key``, ``api-key-2``, ``api-key-3``..."""
    if slot < 1:
        msg = f'a key slot is 1 or more, not {slot}'
        raise ValueError(msg)
    return API_KEY_USERNAME if slot == 1 else f'{API_KEY_USERNAME}-{slot}'


def _slot_key(host: str, slot: int) -> str:
    """The secret stored in *slot* for *host*; :class:`NoCredential` naming why when there is none."""
    return credential_for(host, username=slot_username(slot))[1]


def stored_slots(host: str) -> int:
    """How many key slots *host* holds, counted up to the first missing one."""
    slot = 0
    while True:
        try:
            _slot_key(host, slot + 1)
        except NoCredential:
            return slot
        slot += 1


def login_command(provider: str, task: str | None = None) -> str:
    """The command that stores *provider*'s key, or *task*'s own -- the remedy a missing-key refusal names."""
    return f'python -m ai_lab.auth login {provider}' + (f' --task {task}' if task else '')


def user_providers_file() -> Path:
    """Where a user's own provider rows live (it need not exist)."""
    return config_root(APP) / 'providers.toml'


def _rows(text: str, origin: str) -> dict[str, Provider]:
    table = tomllib.loads(text)
    for name, row in table.items():
        _refuse_retired(name, '', origin)
        if _RETIRED_KEY_FIELD in row:
            msg = (
                f'provider {name!r} in {origin} names {_RETIRED_KEY_FIELD!r}, retired: keys are no longer read from '
                f'the environment. Rename it to key_name (same value), then store the key: {login_command(name)}'
            )
            raise Unsupported(msg)
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
    for row in table.values():
        twin = table.get(row.api_provider)
        if row.api_provider and (twin is None or twin.auth != 'api'):
            msg = f'provider {row.name!r} names api_provider {row.api_provider!r}, which is not an api row'
            raise Unsupported(msg)
    return table


class Client:
    """One provider and one model. ``post``/``run`` are injectable so a test needs no network."""

    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        effort: str = '',
        task: str | None = None,
        post: Callable[..., dict] = transport.post,
        run: Callable[..., transport.CliResult] = transport.run,
    ) -> None:
        """Bind ``provider``, ``model`` and the wire serving that model; refuse an HTTP provider with no model."""
        if task is not None:
            task_username(task)  # an undeclared task is refused here, not at the first call
        self.wire: Wire = WIRES[provider.wire_for(model)]
        if not model and not self.wire.cli:
            msg = f'{provider.name} needs a model: connect("{provider.name}:<model>")'
            raise Unsupported(msg)
        if effort and self.wire.effort is None:
            msg = (
                f'{provider.name}:{model} takes no effort (wire {self.wire.name}); the wires that do are '
                f'{sorted(n for n, w in WIRES.items() if w.effort)}'
            )
            raise Unsupported(msg)
        self.provider = provider
        self.model = model
        self.effort = effort
        self.task = task
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
            raw, slot = self._send(lambda key: self.wire.decide(request, self.model, self.provider.endpoint, key))
            answers, basis, usage = self.wire.parse_decision(raw, request), Basis.NATIVE, self._usage(raw)
        else:
            value, usage, slot = self._respond_value(emulate.as_response_request(request))
            answers, basis = emulate.parse_answers(value, request.questions), Basis.STATED
        latency = time.perf_counter() - start
        return Decision(answers, basis, self.provider.name, self.model, latency, usage, slot)

    def respond(self, request: ResponseRequest) -> Response:
        """An object of the request's schema."""
        self._require_vision(has_image=request.has_image)
        start = time.perf_counter()
        value, usage, slot = self._respond_value(request)
        return Response(value, self.provider.name, self.model, time.perf_counter() - start, usage, slot)

    def _respond_value(self, request: ResponseRequest) -> tuple[Mapping, Usage | None, int]:
        if self.wire.respond is None:
            msg = (
                f'{self.provider.name} is a decision endpoint and cannot respond in a schema; '
                f'use a provider whose wire responds ({sorted(n for n, w in WIRES.items() if w.respond)})'
            )
            raise Unsupported(msg)
        raw, slot = self._send(lambda key: self.wire.respond(request, self.model, self.provider.endpoint, key))
        return self.wire.parse_response(raw), self._usage(raw), slot

    def _usage(self, raw: object) -> Usage | None:
        return None if self.wire.parse_usage is None else self.wire.parse_usage(raw)

    def _require_vision(self, *, has_image: bool) -> None:
        if has_image and not self.provider.vision:
            msg = f'{self.provider.name} does not read images; drop the Image parts or pick one that does'
            raise Unsupported(msg)

    def _send(self, build: Callable[[str], transport.HttpCall | transport.CliCall]) -> tuple[object, int]:
        """``build(key)``'s reply and the key slot that served it, moving on a slot at an account failure."""
        if not self.provider.keyed:
            return self._dispatch(build('')), 0
        host, failures, last = self.provider.host, [], None
        if self.task is not None:
            try:
                key = api_key(host, task=self.task)
            except NoCredential as missing:
                msg = f'{self.provider.name} has no {self.task} task key for {host} ({missing}); run: '
                raise Unsupported(msg + login_command(self.provider.name, self.task)) from None
            return self._dispatch(build(key)), 1
        slot = 0
        while True:
            slot += 1
            try:
                key = _slot_key(host, slot)
            except NoCredential as missing:
                if slot == 1:
                    msg = f'{self.provider.name} has no key stored for {host} ({missing}); run: '
                    raise Unsupported(msg + login_command(self.provider.name)) from None
                break
            try:
                return self._dispatch(build(key)), slot
            except ProviderError as error:
                if not error.kind:
                    raise
                failures.append(f'slot {slot}: {error.kind}')
                last = error
        msg = f'{self.provider.name}: every key slot for {host} failed ({", ".join(failures)}); last: {last}'
        raise ProviderError(msg, kind=last.kind) from last

    def _dispatch(self, call: transport.HttpCall | transport.CliCall) -> object:
        if self.provider.auth == 'subscription' and os.environ.get('SLURM_JOB_ID'):
            msg = (
                f'{self.provider.name} uses subscription auth, which is local-only and refused on a Slurm '
                f'compute node; use the API provider {self.provider.api_provider!r} there'
            )
            raise Unsupported(msg)
        if self.effort:
            call = self.wire.effort(call, self.effort)
        if isinstance(call, transport.CliCall):
            return self._run(call)
        return self._post(call)


def connect(spec: str, *, task: str | None = None, **options: Callable) -> Client:
    """``'openai:gpt-6-luna'``, ``'openai:gpt-x@high'``, ``'claude_code'``..."""
    spec, _, effort = spec.partition('@')
    name, _, model = spec.partition(':')
    _refuse_retired(name, model, 'connect')
    table = providers()
    if name not in table:
        msg = f'unknown provider {name!r}; known: {sorted(table)} (add one in {user_providers_file()})'
        raise Unsupported(msg)
    return Client(table[name], model, effort=effort, task=task, **options)
