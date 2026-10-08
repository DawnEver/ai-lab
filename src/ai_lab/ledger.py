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

__all__ = ['Ledger', 'Mode', 'Recorded']

_ANSWERS = {cls.__name__: cls for cls in (PredicateAnswer, ChoiceAnswer, ScoreAnswer, Refusal)}


class Mode(StrEnum):
    RECORD = 'record'
    REPLAY = 'replay'


def _key(verb: str, client: str, request: object) -> str:
    return hashlib.sha256(canonical([verb, client, request]).encode('utf-8')).hexdigest()


def _answer_to(a: Answer) -> dict:
    return {'kind': type(a).__name__, **asdict(a)}


def _answer_from(d: dict) -> Answer:
    d = dict(d)
    return _ANSWERS[d.pop('kind')](**d)


class Ledger:
    """A JSONL file of entries; the file is the only state, so two runs can share one."""

    def __init__(self, path: str | Path, mode: Mode | str = Mode.RECORD) -> None:
        self.path = Path(path)
        self.mode = Mode(mode)
        self._entries: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text('utf-8').splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry['key']] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def wrap(self, client: object) -> Recorded:
        return Recorded(client, self)

    def _get(self, key: str, what: str) -> dict | None:
        if key in self._entries:
            return self._entries[key]
        if self.mode is Mode.REPLAY:
            raise ReplayMiss(f'{self.path} holds no entry for {what}; re-run with mode="record"')
        return None

    def _put(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as out:
            out.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self._entries[entry['key']] = entry


class Recorded:
    """A client seen through a ledger; same verbs, same answers, served from the file when known."""

    def __init__(self, client: object, ledger: Ledger) -> None:
        self.client = client
        self.ledger = ledger
        self.name = client.name

    def decide(self, request: DecisionRequest) -> Decision:
        key = _key('decide', self.name, request)
        entry = self.ledger._get(key, f'decide on {self.name}')
        if entry is None:
            d = self.client.decide(request)
            entry = {
                'key': key,
                'verb': 'decide',
                'provider': d.provider,
                'model': d.model,
                'basis': d.basis.value,
                'latency_s': d.latency_s,
                'answers': [_answer_to(a) for a in d.answers.values()],
            }
            self.ledger._put(entry)
            return d
        answers = {a.name: a for a in map(_answer_from, entry['answers'])}
        return Decision(answers, Basis(entry['basis']), entry['provider'], entry['model'], entry['latency_s'])

    def respond(self, request: ResponseRequest) -> Response:
        key = _key('respond', self.name, request)
        entry = self.ledger._get(key, f'respond on {self.name}')
        if entry is None:
            r = self.client.respond(request)
            entry = {
                'key': key,
                'verb': 'respond',
                'provider': r.provider,
                'model': r.model,
                'latency_s': r.latency_s,
                'value': r.value,
            }
            self.ledger._put(entry)
            return r
        return Response(entry['value'], entry['provider'], entry['model'], entry['latency_s'])
