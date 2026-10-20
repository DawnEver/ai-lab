"""Every call, keyed by exactly what was asked, so a run can be replayed without a network.

The key is the SHA-256 of the verb, the client name and the canonical request (images by digest),
so an entry answers only the request that produced it. ``RECORD`` serves a hit and records a miss;
``REPLAY`` serves a hit and raises :class:`ReplayMiss` on a miss -- it never reaches a model.

An entry carries :data:`LEDGER_SCHEMA_VERSION`. An older entry is UPGRADED on read, one declared step
per schema (:data:`UPGRADES`), and the file is written back at the current schema; only a schema this
reader does not know is refused.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from ai_lab import retired
from ai_lab.answers import Answer, Basis, ChoiceAnswer, Decision, PredicateAnswer, Refusal, Response, ScoreAnswer
from ai_lab.errors import ReplayMiss, Unsupported
from ai_lab.spec import DecisionRequest, ResponseRequest, canonical
from ai_lab.usage import Usage

__all__ = ['LEDGER_SCHEMA_VERSION', 'UPGRADES', 'Ledger', 'Mode', 'Recorded', 'read_entry']

_ANSWERS = {cls.__name__: cls for cls in (PredicateAnswer, ChoiceAnswer, ScoreAnswer, Refusal)}


class Mode(StrEnum):
    """``RECORD`` serves hits and records misses; ``REPLAY`` serves hits and raises on a miss."""

    RECORD = 'record'
    REPLAY = 'replay'


def _key(verb: str, client: str, request: DecisionRequest | ResponseRequest) -> str:
    return hashlib.sha256('\n'.join((verb, client, canonical(request))).encode()).hexdigest()


def _v1_to_v2(entry: dict) -> dict:
    """Schema 1 (no ``schema_version``) carried vendor dollars in ``usage``; dollars are derived now."""
    usage = entry.get('usage')
    return {**entry, 'usage': None if usage is None else {k: v for k, v in usage.items() if k != 'cost_usd'}}


#: The schema an entry is written at, and the step that lifts each older schema to the next.
LEDGER_SCHEMA_VERSION: Final = 2
UPGRADES: Final = {1: _v1_to_v2}


def read_entry(entry: dict, origin: str) -> dict:
    """*entry* at :data:`LEDGER_SCHEMA_VERSION`; :class:`Unsupported` for a schema this reader does not know."""
    version = entry.get('schema_version', 1)
    if version not in UPGRADES and version != LEDGER_SCHEMA_VERSION:
        msg = f'{origin} holds a ledger entry at schema {version}; this reader knows 1..{LEDGER_SCHEMA_VERSION}'
        raise Unsupported(msg)
    while version != LEDGER_SCHEMA_VERSION:
        entry = UPGRADES[version](entry)
        version += 1
    for field in entry.get('usage') or ():
        retired.refuse('usage', field, origin)
    return {**entry, 'schema_version': version}


def _usage_to(usage: Usage | None) -> dict | None:
    return None if usage is None else asdict(usage)


def _usage_from(d: dict | None) -> Usage | None:
    return None if d is None else Usage(**d)


def _answer_to(a: Answer) -> dict:
    return {'kind': type(a).__name__, **asdict(a)}


def _answer_from(d: dict) -> Answer:
    d = dict(d)
    return _ANSWERS[d.pop('kind')](**d)


class Ledger:
    """A JSONL file of entries; the file is the only state, so two runs can share one."""

    def __init__(self, path: str | Path, mode: Mode | str = Mode.RECORD) -> None:
        """Open ``path`` (it need not exist yet) in ``mode``."""
        self.path = Path(path)
        self.mode = Mode(mode)
        self._entries: dict[str, dict] = {}
        if not self.path.exists():
            return
        stale = False
        for line in self.path.read_text('utf-8').splitlines():
            if line.strip():
                raw = json.loads(line)
                entry = read_entry(raw, str(self.path))
                stale = stale or entry != raw
                self._entries[entry['key']] = entry
        if stale:
            fresh = self.path.with_name(self.path.name + '.upgrading')
            lines = (json.dumps(e, ensure_ascii=False) + '\n' for e in self._entries.values())
            fresh.write_text(''.join(lines), 'utf-8')
            fresh.replace(self.path)

    def __len__(self) -> int:
        """How many entries the file holds."""
        return len(self._entries)

    def wrap(self, client: object) -> Recorded:
        """``client`` seen through this ledger."""
        return Recorded(client, self)

    def entry(self, key: str, what: str) -> dict | None:
        """The entry recorded under ``key``; ``None`` on a miss in ``RECORD``, a refusal in ``REPLAY``."""
        if key in self._entries:
            return self._entries[key]
        if self.mode is Mode.REPLAY:
            msg = f'{self.path} holds no entry for {what}; re-run with mode="record"'
            raise ReplayMiss(msg)
        return None

    def append(self, entry: dict) -> None:
        """Write ``entry`` to the file and serve it from now on."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as out:
            out.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self._entries[entry['key']] = entry


class Recorded:
    """A client seen through a ledger; same verbs, same answers, served from the file when known."""

    def __init__(self, client: object, ledger: Ledger) -> None:
        """``client``, answered from ``ledger`` when the exact request is known."""
        self.client = client
        self.ledger = ledger
        self.name = client.name

    def decide(self, request: DecisionRequest) -> Decision:
        """The recorded decision for this request, else the client's (recorded in ``RECORD`` mode)."""
        key = _key('decide', self.name, request)
        entry = self.ledger.entry(key, f'decide on {self.name}')
        if entry is None:
            d = self.client.decide(request)
            entry = {
                'schema_version': LEDGER_SCHEMA_VERSION,
                'key': key,
                'verb': 'decide',
                'client': self.name,
                'at': datetime.now(UTC).isoformat(),
                'provider': d.provider,
                'model': d.model,
                'basis': d.basis.value,
                'latency_s': d.latency_s,
                'usage': _usage_to(d.usage),
                'key_slot': d.key_slot,
                'answers': [_answer_to(a) for a in d.answers.values()],
            }
            self.ledger.append(entry)
            return d
        answers = {a.name: a for a in map(_answer_from, entry['answers'])}
        return Decision(
            answers,
            Basis(entry['basis']),
            entry['provider'],
            entry['model'],
            entry['latency_s'],
            _usage_from(entry.get('usage')),
            entry.get('key_slot', 0),
        )

    def respond(self, request: ResponseRequest) -> Response:
        """The recorded response for this request, else the client's (recorded in ``RECORD`` mode)."""
        key = _key('respond', self.name, request)
        entry = self.ledger.entry(key, f'respond on {self.name}')
        if entry is None:
            r = self.client.respond(request)
            entry = {
                'schema_version': LEDGER_SCHEMA_VERSION,
                'key': key,
                'verb': 'respond',
                'client': self.name,
                'at': datetime.now(UTC).isoformat(),
                'provider': r.provider,
                'model': r.model,
                'latency_s': r.latency_s,
                'usage': _usage_to(r.usage),
                'key_slot': r.key_slot,
                'value': r.value,
            }
            self.ledger.append(entry)
            return r
        return Response(
            entry['value'],
            entry['provider'],
            entry['model'],
            entry['latency_s'],
            _usage_from(entry.get('usage')),
            entry.get('key_slot', 0),
        )
