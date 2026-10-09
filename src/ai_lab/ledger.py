"""Every call, keyed by exactly what was asked, so a run can be replayed without a network.

The key is the SHA-256 of the verb, the client name and the canonical request (images by digest),
so an entry answers only the request that produced it. ``RECORD`` serves a hit and records a miss;
``REPLAY`` serves a hit and raises :class:`ReplayMiss` on a miss -- it never reaches a model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path

from ai_lab.answers import Answer, Basis, ChoiceAnswer, Decision, PredicateAnswer, Refusal, Response, ScoreAnswer
from ai_lab.errors import ReplayMiss
from ai_lab.spec import DecisionRequest, ResponseRequest, canonical
from ai_lab.usage import Usage

__all__ = ['Ledger', 'Mode', 'Recorded']

_ANSWERS = {cls.__name__: cls for cls in (PredicateAnswer, ChoiceAnswer, ScoreAnswer, Refusal)}


class Mode(StrEnum):
    """``RECORD`` serves hits and records misses; ``REPLAY`` serves hits and raises on a miss."""

    RECORD = 'record'
    REPLAY = 'replay'


def _key(verb: str, client: str, request: DecisionRequest | ResponseRequest) -> str:
    return hashlib.sha256('\n'.join((verb, client, canonical(request))).encode()).hexdigest()


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
        if self.path.exists():
            for line in self.path.read_text('utf-8').splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry['key']] = entry

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
                'key': key,
                'verb': 'decide',
                'client': self.name,
                'provider': d.provider,
                'model': d.model,
                'basis': d.basis.value,
                'latency_s': d.latency_s,
                'usage': _usage_to(d.usage),
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
        )

    def respond(self, request: ResponseRequest) -> Response:
        """The recorded response for this request, else the client's (recorded in ``RECORD`` mode)."""
        key = _key('respond', self.name, request)
        entry = self.ledger.entry(key, f'respond on {self.name}')
        if entry is None:
            r = self.client.respond(request)
            entry = {
                'key': key,
                'verb': 'respond',
                'client': self.name,
                'provider': r.provider,
                'model': r.model,
                'latency_s': r.latency_s,
                'usage': _usage_to(r.usage),
                'value': r.value,
            }
            self.ledger.append(entry)
            return r
        return Response(
            entry['value'], entry['provider'], entry['model'], entry['latency_s'], _usage_from(entry.get('usage'))
        )
