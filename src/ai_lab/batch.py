"""BATCHED CALLS THAT OUTLIVE THE PROCESS: a program replayed until every call it makes has a reply.

User ruling 2026-11-04: review model calls go through the vendor's Message Batches API BY DEFAULT (half the price,
asynchronous up to 24 h, usually minutes; prompt caching stacks with it); the synchronous path stays an option.

A PROGRAM is any function of one ``call`` (``ai_lab.call.answer``'s shape: one request dict in, one reply out).
:meth:`Batched.settle` runs it with a ``call`` that answers from what earlier batches returned and, for a request
it has no reply for yet, records it and raises :class:`NotYet`. A program whose units each catch :class:`NotYet`
(every reviewer, every finding) lets all of them queue in ONE pass; whatever was queued is submitted as one batch
per client, polled with backoff, and the program is replayed from the top -- served from the results this time,
one step further. A conversation of N turns is N batches; N reviewers in a turn are one.

THE STATE IS SMALL AND HOLDS NO SECRET (:class:`State`): a seed and, per submitted batch, its id, the key SLOT it
was submitted under, the client it was built from and the custom ids it carries. A custom id is the request's
:func:`digest`, so a replayed program asks for exactly the requests it asked for before and finds their replies
by content, never by position; nothing a model wrote is stored -- results are read back from the vendor (kept
29 days). The program must be DETERMINISTIC given the state: the randomness it needs comes from
:meth:`State.rng` (seeded, per label), never from the system.

THE WAIT IS BOUNDED TWICE. ``wait_s`` is THIS process's budget: when it is spent with a batch still running,
:meth:`~Batched.settle` returns :class:`Pending` (the state saved, nothing printed as a result) and the caller
exits :data:`PENDING_EXIT`; the next run with the same state resumes polling and never resubmits. The vendor's
own expiry (24 h) bounds the whole: a batch still running :data:`EXPIRY_GRACE` past it is canceled and its
requests taken as expired. An expired or canceled request is answered SYNCHRONOUSLY under ``on_expiry = "sync"``
and is a ``timeout`` (canceled: unclassified) failure under ``"fail"`` (:data:`ON_EXPIRY`).

A request that cannot be batched -- a ``decide``, a provider whose wire declares no batch endpoint -- is answered at
once by ``answer``. :class:`Direct` is ``batch = false``: every call answered at once, never pending.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import secrets
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, TypeVar

from ai_lab.call import answer as _answer
from ai_lab.call import client_spec, failure_reply, parsed, response_reply
from ai_lab.client import Ticket, connect
from ai_lab.errors import BatchLapsed, SpendCapReached, ProviderError, Unsupported

__all__ = [
    'BATCH_BY_DEFAULT', 'EXPIRY_GRACE', 'ON_EXPIRY', 'PENDING_EXIT', 'POLL_FIRST_S', 'POLL_MAX_S', 'WAIT_S',
    'Batched', 'Direct', 'NotYet', 'Pending', 'State', 'add_arguments', 'digest', 'from_arguments', 'runner',
]  # fmt: skip

#: Whether a call is batched when the caller says nothing (user ruling 2026-11-04): yes.
BATCH_BY_DEFAULT: Final = True

#: The exit a CLI ends with when its batch is still running at the end of its own wait: EX_TEMPFAIL (sysexits.h),
#: "try again later". Nothing is on stdout; the state names the batch and the next run resumes it.
PENDING_EXIT: Final = 75

#: What an expired or canceled request becomes: answered synchronously, or a failure.
ON_EXPIRY: Final = ('sync', 'fail')

#: The first pause between two polls, doubling up to :data:`POLL_MAX_S`: most batches end within minutes.
POLL_FIRST_S: Final = 15.0
POLL_MAX_S: Final = 300.0

#: How long past a batch's own ``expires_at`` it is still waited for before it is canceled and taken as expired.
EXPIRY_GRACE: Final = timedelta(hours=1)

#: A process's default wait: the vendor's 24 h expiry and its grace -- with no budget of its own, a run waits it out.
WAIT_S: Final = (timedelta(hours=24) + EXPIRY_GRACE).total_seconds()

_STATE_VERSION: Final = 1

T = TypeVar('T')


class NotYet(Exception):
    """A call whose reply no batch has returned yet: it is queued, and the program is replayed once it has."""


def digest(request: Mapping) -> str:
    """The request's identity: sha256 of its canonical JSON, 64 hex characters -- a valid batch custom id."""
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


@dataclass
class State:
    """What a run leaves for the next: a seed and every batch it submitted (never a reply, never a key).

    ``path`` ``None`` is a state no other run reads (kept in memory).
    """

    path: Path | None
    seed: str
    batches: list[dict] = field(default_factory=list)

    @classmethod
    def fresh(cls, path: Path | None) -> State:
        """A new state with a new seed."""
        return cls(path, secrets.token_hex(16))

    @classmethod
    def load(cls, path: str | Path) -> State:
        """The state at *path*, a fresh one when the file does not exist; ``ValueError`` when it is not a state."""
        path = Path(path)
        if not path.is_file() or not path.read_text(encoding='utf-8').strip():
            return cls.fresh(path)
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as bad:
            msg = f'{path} is not a batch state: {bad}'
            raise ValueError(msg) from None
        if (
            not isinstance(data, dict)
            or data.get('version') != _STATE_VERSION
            or not isinstance(data.get('seed'), str)
            or not isinstance(data.get('batches'), list)
            or not all(_is_batch(one) for one in data['batches'])
        ):
            msg = f'{path} is not a batch state of version {_STATE_VERSION}'
            raise ValueError(msg)
        return cls(path, data['seed'], data['batches'])

    def save(self) -> None:
        """Write the state where it was loaded from (atomically); a path-less state is not written."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({'version': _STATE_VERSION, 'seed': self.seed, 'batches': self.batches}, indent=1)
        handle, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix='.tmp')
        with os.fdopen(handle, 'w', encoding='utf-8') as out:
            out.write(text + '\n')
        Path(temporary).replace(self.path)

    def rng(self, label: str) -> random.Random:
        """A generator seeded by the state and *label*: the same in every replay, different per label."""
        return random.Random(f'{self.seed}:{label}')  # noqa: S311 -- planning and fence nonces, not secrets


def _is_batch(one: object) -> bool:
    keys = {'id', 'slot', 'expires_at', 'spec', 'options', 'ids', 'submitted_at', 'charged'}
    return isinstance(one, dict) and keys <= set(one) and isinstance(one['ids'], list)


@dataclass(frozen=True)
class Pending:
    """The batches still running when a run's own wait was spent."""

    ids: tuple[str, ...]
    submitted_at: str
    expires_at: str

    def line(self) -> str:
        """One line a head's status can carry."""
        return (
            f'model batch {", ".join(self.ids)} submitted {self.submitted_at}, results pending '
            f'(expires {self.expires_at}); the next run resumes it, nothing is resubmitted'
        )


#: A served marker: the request lapsed and is to be answered synchronously when the program asks for it again.
_SYNC: Final = object()


class Direct:
    """``batch = false``: every call is answered at once; never :class:`Pending`."""

    def __init__(self, *, answer: Callable[[Mapping], Mapping] = _answer) -> None:
        """*answer* makes each call (``ai_lab.call.answer`` or a stand-in)."""
        self.answer = answer

    def rng(self, _label: str) -> random.Random:
        """The system's generator: a direct run is never replayed."""
        return random.SystemRandom()

    def settle(self, program: Callable[[Callable[[Mapping], Mapping]], T]) -> T:
        """*program*'s result, its calls answered at once."""
        return program(self.answer)


class Batched:
    """Runs a program over the vendor's batch endpoint, replaying it until it is done or the wait is spent."""

    def __init__(
        self, state: State, *, connect: Callable[..., object] = connect,
        answer: Callable[[Mapping], Mapping] = _answer, clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep, wait_s: float = WAIT_S, on_expiry: str = 'sync',
    ) -> None:  # fmt: skip
        """*wait_s* is this process's budget for waiting on a batch; *on_expiry* one of :data:`ON_EXPIRY`."""
        if on_expiry not in ON_EXPIRY:
            msg = f'on_expiry {on_expiry!r} is not one of {ON_EXPIRY}'
            raise ValueError(msg)
        self.state, self.connect, self.answer = state, connect, answer
        self.clock, self.sleep, self.wait_s, self.on_expiry = clock, sleep, wait_s, on_expiry
        self._served: dict[str, object] = {}
        self._read: set[str] = set()
        self._wanted: dict[str, Mapping] = {}

    def rng(self, label: str) -> random.Random:
        """The state's seeded generator for *label* (:meth:`State.rng`)."""
        return self.state.rng(label)

    def call(self, request: Mapping) -> Mapping:
        """*request*'s reply when a batch returned it or it cannot be batched; else queue it, :class:`NotYet`."""
        key = digest(request)
        if key in self._served:
            served = self._served[key]
            if served is _SYNC:
                served = self._served[key] = self.answer(request)
            return served
        verb, _typed = parsed(request)
        if verb != 'respond' or not self._batchable(request):
            reply = self._served[key] = self.answer(request)
            return reply
        self._wanted[key] = request
        raise NotYet

    def _batchable(self, request: Mapping) -> bool:
        spec, options = client_spec(request)
        try:
            return bool(self.connect(spec, **options).can_batch)
        except (Unsupported, ValueError):
            return False  # answer() says why, as the call's own reply

    def settle(self, program: Callable[[Callable[[Mapping], Mapping]], T]) -> T | Pending:
        """*program*'s result once every call it makes has a reply, or :class:`Pending` when the wait is spent."""
        deadline = self.clock() + timedelta(seconds=self.wait_s)
        pause = POLL_FIRST_S
        while True:
            running = self._read_ended()
            if running:
                now = self.clock()
                if now >= deadline:
                    return Pending(
                        tuple(b['id'] for b in running), min(b['submitted_at'] for b in running),
                        max(b['expires_at'] for b in running),
                    )  # fmt: skip
                self.sleep(min(pause, (deadline - now).total_seconds()))
                pause = min(pause * 2, POLL_MAX_S)
                continue
            self._wanted = {}
            try:
                result = program(self.call)
            except NotYet:
                result = None
            if not self._wanted:
                return result
            self._submit(self._wanted)
            pause = POLL_FIRST_S

    def _submit(self, wanted: Mapping[str, Mapping]) -> None:
        """One batch per client (provider, model, task, tag, budget) of the *wanted* requests, recorded."""
        groups: dict[str, tuple[str, dict, dict]] = {}
        for key, request in wanted.items():
            spec, options = client_spec(request)
            group = groups.setdefault(json.dumps([spec, options], sort_keys=True), (spec, options, {}))
            group[2][key] = parsed(request)[1]
        for spec, options, requests in groups.values():
            try:
                ticket = self.connect(spec, **options).submit(requests)
            except (Unsupported, ProviderError, SpendCapReached) as failed:
                for key in requests:
                    self._served[key] = failure_reply(failed)
                continue
            self.state.batches.append({
                'id': ticket.id, 'slot': ticket.slot, 'expires_at': ticket.expires_at, 'spec': spec,
                'options': options, 'ids': sorted(requests), 'charged': False,
                'submitted_at': self.clock().isoformat(timespec='seconds'),
            })  # fmt: skip
            self.state.save()

    def _read_ended(self) -> list[dict]:
        """Read every ended batch not read yet into the served replies; the batches still running."""
        running = []
        for batch in self.state.batches:
            if batch['id'] in self._read:
                continue
            client = self.connect(batch['spec'], **batch['options'])
            ticket = Ticket(batch['id'], batch['slot'], batch['expires_at'])
            try:
                ended = client.ended(ticket)
            except ProviderError:
                ended = False  # a failed poll is no news: the expiry below still bounds the wait
            if ended:
                self._take(batch, client.results(ticket, charge=not batch['charged']))
            elif self._lapsed(batch):
                with contextlib.suppress(ProviderError):  # the vendor expires it anyway
                    client.cancel(ticket)
                expired = BatchLapsed(f'batch {batch["id"]} ran past its expiry; taken as expired', lapsed='expired')
                self._take(batch, dict.fromkeys(batch['ids'], expired))
            else:
                running.append(batch)
        return running

    def _lapsed(self, batch: Mapping) -> bool:
        try:
            expires = datetime.fromisoformat(str(batch['expires_at']))
        except ValueError:
            return False
        return self.clock() > expires + EXPIRY_GRACE

    def _take(self, batch: dict, results: Mapping[str, object]) -> None:
        for key in batch['ids']:
            got = results.get(key)
            if got is None:
                got = ProviderError(f'batch {batch["id"]} returned no result for a request it was sent')
            if isinstance(got, BatchLapsed) and self.on_expiry == 'sync':
                self._served[key] = _SYNC
            elif isinstance(got, Exception):
                self._served[key] = failure_reply(got)
            else:
                self._served[key] = response_reply(got)
        self._read.add(batch['id'])
        if not batch['charged']:
            batch['charged'] = True
            self.state.save()


def runner(
    *, batch: bool, state: str | Path | None, wait_s: float, on_expiry: str,
    answer: Callable[[Mapping], Mapping] = _answer, connect: Callable[..., object] = connect,
) -> Direct | Batched:  # fmt: skip
    """The runner a command line asks for: :class:`Direct` without *batch*, else :class:`Batched` over *state*."""
    if not batch:
        return Direct(answer=answer)
    loaded = State.load(state) if state else State.fresh(None)
    return Batched(loaded, connect=connect, answer=answer, wait_s=wait_s, on_expiry=on_expiry)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags every batched command line takes: ``--[no-]batch``, ``--batch-state``, ``--wait``, ``--on-expiry``."""
    parser.add_argument(
        '--batch', action=argparse.BooleanOptionalAction, default=BATCH_BY_DEFAULT,
        help='make the model calls through the vendor batch endpoint (half price, asynchronous); --no-batch: at once',
    )  # fmt: skip
    parser.add_argument(
        '--batch-state', metavar='FILE',
        help=f'keep the batch state here: a run that ends pending (exit {PENDING_EXIT}) is resumed by the next run '
        'given the same file, never resubmitted',
    )  # fmt: skip
    parser.add_argument(
        '--wait', type=float, default=WAIT_S, metavar='SECONDS',
        help=f'how long this run waits on a running batch before it exits {PENDING_EXIT} (default: its expiry)',
    )  # fmt: skip
    parser.add_argument(
        '--on-expiry', choices=ON_EXPIRY, default=ON_EXPIRY[0],
        help='an expired or canceled batch request: answered at once (sync) or a timeout failure (fail)',
    )  # fmt: skip


def from_arguments(
    args: argparse.Namespace, *, answer: Callable[[Mapping], Mapping] = _answer,
    connect: Callable[..., object] = connect,
) -> Direct | Batched:  # fmt: skip
    """The runner :func:`add_arguments`' flags ask for."""
    return runner(batch=args.batch, state=args.batch_state, wait_s=args.wait, on_expiry=args.on_expiry,
                  answer=answer, connect=connect)  # fmt: skip
