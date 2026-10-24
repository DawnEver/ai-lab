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
inferred from an empty ``key_name``. ai-lab is the pay-per-token API layer: a row is a keyed ``api``
account or a keyless ``local`` server, and every retired spelling is refused by name (:mod:`ai_lab.retired`).

ONE PROVIDER IS ONE ACCOUNT KIND OF ONE VENDOR (user ruling 2026-10-16). A vendor endpoint that only
some models of the account serve (OpenAI's ``/decisions``) is not a second row: the row declares
``model_wires = {model = wire}`` and :meth:`Provider.wire_for` picks the wire by the MODEL, as data.

THIS TABLE IS THE ONE PROVIDER CATALOGUE (plan 2026-10-20). A row is reached at its ``url`` (an
origin) plus a ``<protocol>_path`` for every protocol (:data:`PROTOCOLS`) the account serves, so a
second protocol into one account (DeepSeek's Anthropic-compatible endpoint beside its OpenAI-compatible
one) is a path of the same row and ``connect(spec, protocol=...)`` picks the wire by protocol. Its
``models`` map names the model each Claude-class ROLE (:data:`ROLES`) resolves to -- the data a host
launcher projects from; ``python -m ai_lab.providers --json`` publishes the whole table.

AN API ACCOUNT'S KEYS ARE ORDERED SLOTS: usernames ``api-key``, ``api-key-2``, ``api-key-3``... under
the endpoint host, read up to the first missing slot. A call starts on slot 1 and moves to the next
slot when it fails at the ACCOUNT level (``ProviderError.kind``: exhausted, auth, rate limit); when
every slot fails, the error lists each slot's failure kind -- never a key. The slot that served a
call is on its :class:`~ai_lab.answers.Decision` / :class:`~ai_lab.answers.Response`.

A TASK OWNS ITS KEYS (user rulings 2026-10-19). ``connect(spec, task='optimization')`` walks that task's
slots -- usernames ``api-key-<task>``, ``api-key-<task>-2``... under the same host
(:func:`lab_commons.credentials.key_username`) -- in the same fall-over order, and only those: never another
task's slot, never the default's. A task with no slot 1 is refused naming ``login <provider> --task <task>``.
"""

from __future__ import annotations

import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit

from lab_commons.credentials import NoCredential, api_key, key_username
from lab_commons.paths import config_root

from ai_lab import budget, emulate, retired, transport
from ai_lab.answers import Basis, Decision, Response
from ai_lab.errors import ACCOUNT_KINDS, DailyTokenCapReached, ProviderError, Unsupported
from ai_lab.spec import DecisionRequest, ResponseRequest
from ai_lab.usage import Usage
from ai_lab.wire import WIRES
from ai_lab.wire.base import Wire

__all__ = [
    'APP',
    'AUTH_KINDS',
    'PROTOCOLS',
    'ROLES',
    'Client',
    'Provider',
    'catalogue',
    'connect',
    'packaged_providers_file',
    'stored_slots',
    'user_providers_file',
]

#: The application name the family's path helpers resolve this package's roots under.
APP = 'ai_lab'


#: How a provider authenticates: a keyed per-token ``api`` account, or a keyless ``local`` server you run.
AUTH_KINDS = frozenset({'api', 'local'})

#: The API families a wire speaks; a row serves one at its ``<protocol>_path``.
PROTOCOLS = frozenset({'openai', 'anthropic', 'gemini', 'typesafe'})

#: The Claude-class roles a row's ``models`` map may name; ``base`` is required and the default of the rest.
ROLES = frozenset({'base', 'fable', 'opus', 'sonnet', 'haiku', 'subagent', 'codex'})


@dataclass(frozen=True, slots=True)
class Provider:
    """One vendor account: its default wire, where each protocol it serves is reached, and its key.

    ``key_name`` is the conventional name of the vendor's key (``OPENAI_API_KEY``); ``''`` is keyless.
    It is never looked up in the environment: the key itself lives in the git credential store under
    :attr:`host`, and the name only lets ``python -m ai_lab.auth login --from-file`` import a key file.
    """

    name: str
    wire: str
    #: The origin (scheme and host, no path); a protocol's base URL is this plus its ``*_path``.
    url: str
    #: One of :data:`AUTH_KINDS`.
    auth: str
    key_name: str = ''
    vision: bool = False
    #: Who bills this provider's tokens, as the price catalogs name it; ``''`` is unmetered.
    vendor: str = ''
    #: ``{model: wire}`` -- a model of this account served by another wire than :attr:`wire`.
    model_wires: Mapping[str, str] = field(default_factory=dict)
    #: The path of each protocol this account serves; ``''`` is a protocol it does not.
    openai_path: str = ''
    anthropic_path: str = ''
    gemini_path: str = ''
    typesafe_path: str = ''
    #: ``{role: model}`` over :data:`ROLES` -- the model each Claude-class role resolves to.
    models: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse an unknown wire or auth kind, a wire with no path, a url with a path, a malformed role map."""
        where = f'provider {self.name!r}'
        retired.refuse('auth', self.auth, where)
        if self.auth not in AUTH_KINDS:
            msg = f'{where} names unknown auth {self.auth!r}; the kinds are {sorted(AUTH_KINDS)}'
            raise Unsupported(msg)
        if urlsplit(self.url).path.strip('/'):
            msg = f'{where}: url {self.url!r} must be an origin; put the path in <protocol>_path'
            raise Unsupported(msg)
        for wire in (self.wire, *self.model_wires.values()):
            retired.refuse('wire', wire, where)
            if wire not in WIRES:
                msg = f'{where} names unknown wire {wire!r}; the wires are {sorted(WIRES)}'
                raise Unsupported(msg)
            protocol = WIRES[wire].protocol
            if protocol not in self.protocols:
                msg = f'{where}: wire {wire!r} speaks {protocol!r}, so the row needs {protocol}_path'
                raise Unsupported(msg)
        if unknown := sorted(set(self.models) - ROLES):
            msg = f'{where}: models names unknown role {unknown[0]!r}; the roles are {sorted(ROLES)}'
            raise Unsupported(msg)
        if self.models and 'base' not in self.models:
            msg = f"{where}: a models map needs 'base', the default of every other role"
            raise Unsupported(msg)

    @property
    def protocols(self) -> frozenset[str]:
        """The protocols this account serves: those with a path."""
        return frozenset(p for p in PROTOCOLS if self.path(p))

    def path(self, protocol: str) -> str:
        """*protocol*'s declared path on this account; ``''`` when it does not serve it."""
        return {
            'openai': self.openai_path,
            'anthropic': self.anthropic_path,
            'gemini': self.gemini_path,
            'typesafe': self.typesafe_path,
        }.get(protocol, '')

    def endpoint(self, protocol: str) -> str:
        """The base URL of *protocol* on this account; :class:`Unsupported` when it is not served."""
        if protocol not in PROTOCOLS:
            msg = f'unknown protocol {protocol!r}; the protocols are {sorted(PROTOCOLS)}'
            raise Unsupported(msg)
        if protocol not in self.protocols:
            msg = f'{self.name} does not serve protocol {protocol!r}; it serves {sorted(self.protocols)}'
            raise Unsupported(msg)
        return self.url.rstrip('/') + self.path(protocol)

    def wire_for(self, model: str, protocol: str | None = None) -> str:
        """The wire serving *model* (its :attr:`model_wires` entry, else :attr:`wire`), over *protocol* if named.

        A named protocol other than that wire's is served by the one shipped wire of that protocol;
        a protocol several wires speak is refused naming them.
        """
        wire = self.model_wires.get(model, self.wire)
        if protocol is None or WIRES[wire].protocol == protocol:
            return wire
        self.endpoint(protocol)  # refuses an unknown or unserved protocol by name
        speaking = sorted(n for n, w in WIRES.items() if w.protocol == protocol)
        if len(speaking) != 1:
            msg = f'{self.name}: protocol {protocol!r} is spoken by {speaking}; declare the model in model_wires'
            raise Unsupported(msg)
        return speaking[0]

    @property
    def keyed(self) -> bool:
        """Whether a call needs a stored key."""
        return bool(self.key_name)

    @property
    def host(self) -> str:
        """The host the key is stored under."""
        return urlsplit(self.url).hostname or ''


def _slot_key(host: str, slot: int, task: str | None = None) -> str:
    """The secret in *task*'s (else the default's) *slot* for *host*; :class:`NoCredential` naming why."""
    return api_key(host, task=task, slot=slot)


def stored_slots(host: str, task: str | None = None) -> int:
    """How many of *task*'s (else the default's) key slots *host* holds, up to the first missing one."""
    slot = 0
    while True:
        try:
            _slot_key(host, slot + 1, task)
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
        retired.refuse('provider', name, origin)
        for key in row:
            retired.refuse('field', key, f'provider {name!r} in {origin}')
    try:
        return {name: Provider(name=name, **row) for name, row in table.items()}
    except TypeError as error:
        msg = f'a provider row in {origin} has an unknown or missing field: {error}'
        raise Unsupported(msg) from None


def packaged_providers_file() -> Path:
    """The packaged baseline ``providers.toml``."""
    return Path(__file__).with_name('providers.toml')


def catalogue() -> dict[str, Provider]:
    """The packaged table, with every row of :func:`user_providers_file` replacing its namesake."""
    table = _rows(packaged_providers_file().read_text('utf-8'), 'the packaged providers.toml')
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
        task: str | None = None,
        protocol: str | None = None,
        post: Callable[..., dict] = transport.post,
    ) -> None:
        """Bind ``provider``, ``model`` and the wire serving that model; refuse a provider with no model."""
        if task is not None:
            key_username(task)  # an undeclared task is refused here, not at the first call
        self.wire: Wire = WIRES[provider.wire_for(model, protocol)]
        self.endpoint = provider.endpoint(self.wire.protocol)
        if not model:
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
            raw, slot = self._send(lambda key: self.wire.decide(request, self.model, self.endpoint, key))
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
        raw, slot = self._send(lambda key: self.wire.respond(request, self.model, self.endpoint, key))
        return self.wire.parse_response(raw), self._usage(raw), slot

    def _usage(self, raw: object) -> Usage | None:
        return None if self.wire.parse_usage is None else self.wire.parse_usage(raw)

    def _require_vision(self, *, has_image: bool) -> None:
        if has_image and not self.provider.vision:
            msg = f'{self.provider.name} does not read images; drop the Image parts or pick one that does'
            raise Unsupported(msg)

    def _send(self, build: Callable[[str], transport.HttpCall]) -> tuple[object, int]:
        """``build(key)``'s reply and the key slot that served it, moving on a slot at an account failure."""
        if not self.provider.keyed:
            return self._dispatch(build(''), None), 0
        host, failures, last = self.provider.host, [], None
        whose = '' if self.task is None else f'{self.task} task '
        slot = 0
        while True:
            slot += 1
            try:
                key = _slot_key(host, slot, self.task)
            except NoCredential as missing:
                if slot == 1:
                    msg = f'{self.provider.name} has no {whose}key stored for {host} ({missing}); run: '
                    raise Unsupported(msg + login_command(self.provider.name, self.task)) from None
                break
            try:
                return self._dispatch(build(key), budget.fingerprint(key)), slot
            except DailyTokenCapReached as spent:
                if spent.scope == self.provider.name:
                    raise
                failures.append(f'slot {slot}: {spent.scope} daily token cap')
                last = spent
                continue
            except ProviderError as error:
                if error.kind not in ACCOUNT_KINDS:
                    raise
                failures.append(f'slot {slot}: {error.kind}')
                last = error
        msg = f'{self.provider.name}: every {whose}key slot for {host} failed ({", ".join(failures)}); last: {last}'
        if isinstance(last, DailyTokenCapReached):
            raise DailyTokenCapReached(msg, scope=last.scope) from last
        raise ProviderError(msg, kind=last.kind) from last

    def _dispatch(self, call: transport.HttpCall, key: str | None) -> object:
        """Send *call* within its provider's and *key*'s token budgets, and charge what its reply reports."""
        left = budget.remaining(self.provider.name, key)
        if 'max_tokens' in call.body:
            call = replace(call, body={**call.body, 'max_tokens': min(call.body['max_tokens'], left)})
        if self.effort:
            call = self.wire.effort(call, self.effort)
        raw = self._post(call)
        if (usage := self._usage(raw)) is not None:
            budget.charge(self.provider.name, key, usage.input_tokens + usage.output_tokens)
        return raw


def connect(spec: str, *, task: str | None = None, protocol: str | None = None, **options: Callable) -> Client:
    """``'openai:gpt-6-luna'``, ``'openai:gpt-x@high'``, ``'anthropic:<model>'``..."""
    spec, _, effort = spec.partition('@')
    name, _, model = spec.partition(':')
    retired.refuse('provider', name, f'connect({spec!r})')
    table = catalogue()
    if name not in table:
        msg = f'unknown provider {name!r}; known: {sorted(table)} (add one in {user_providers_file()})'
        raise Unsupported(msg)
    return Client(table[name], model, effort=effort, task=task, protocol=protocol, **options)
