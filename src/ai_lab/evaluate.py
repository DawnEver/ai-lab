"""``python -m ai_lab.evaluate DATASET --decider SPEC [...]`` -- score deciders on labelled requests.

A DATASET is JSONL, one recorded request per line with the truth for its questions::

    {"request": <decision_request_to_json>, "labels": {"p0": true, "p1": false}}

A boolean label is "yes is right" (a predicate) or "this one is good" (a score, ranked by its
expected level, and a yes from the upper half of its levels); a string label is the right option of a
choice. Every decider answers the SAME requests, so models, efforts and vendors are compared on
identical questions without re-running
whatever produced them -- a recorded search is a benchmark. With ``--ledger`` every call is served
from that ledger when known, so a re-evaluation costs nothing.

The scorecard: AUC of the decider's signal against the boolean labels (0.5 is chance), precision
and recall of its yes, choice accuracy, refusals, latency and the vendor's token counts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ai_lab.answers import ChoiceAnswer, PredicateAnswer, ScoreAnswer
from ai_lab.client import connect
from ai_lab.ledger import Ledger
from ai_lab.protocols import Decider
from ai_lab.spec import DecisionRequest, decision_request_from_json
from ai_lab.usage import Usage, cost

__all__ = ['Case', 'Scorecard', 'auc', 'evaluate', 'load']


@dataclass(frozen=True)
class Case:
    """One request and the truth for (some of) its questions."""

    request: DecisionRequest
    labels: Mapping[str, bool | str]


def load(path: Path, *, limit: int | None = None) -> list[Case]:
    """The cases of a dataset file, the first ``limit`` when given."""
    cases = []
    for line in Path(path).read_text('utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            cases.append(Case(decision_request_from_json(row['request']), row['labels']))
            if limit is not None and len(cases) >= limit:
                break
    return cases


def auc(signals: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Probability a random positive outranks a random negative (ties half); ``None`` without both."""
    pos = [s for s, y in zip(signals, labels, strict=True) if y]
    neg = [s for s, y in zip(signals, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


@dataclass
class Scorecard:
    """How one decider did on a dataset."""

    decider: str
    calls: int = 0
    seconds: list[float] = field(default_factory=list)
    signals: list[float] = field(default_factory=list)
    truths: list[bool] = field(default_factory=list)
    said_yes: list[bool] = field(default_factory=list)
    choices: list[bool] = field(default_factory=list)
    refused: int = 0
    usage: Usage = field(default_factory=Usage)
    usage_reported: int = 0
    cost_usd: float | None = 0.0

    @property
    def auc(self) -> float | None:
        """Ranking quality of the signal against the boolean labels."""
        return auc(self.signals, self.truths)

    @property
    def precision(self) -> float | None:
        """Share of the yes answers whose label is true."""
        hits = [t for t, y in zip(self.truths, self.said_yes, strict=True) if y]
        return sum(hits) / len(hits) if hits else None

    @property
    def recall(self) -> float | None:
        """Share of the true labels answered yes."""
        hits = [y for t, y in zip(self.truths, self.said_yes, strict=True) if t]
        return sum(hits) / len(hits) if hits else None

    @property
    def base_rate(self) -> float | None:
        """Share of true labels -- what a constant yes would score as precision."""
        return sum(self.truths) / len(self.truths) if self.truths else None


def _read(answer: object, question: object, *, label: bool | str, card: Scorecard) -> None:
    if isinstance(answer, ChoiceAnswer) and isinstance(label, str):
        card.choices.append(answer.choice == label)
        return
    if not isinstance(label, bool):
        return
    if isinstance(answer, PredicateAnswer):
        card.signals.append(answer.probability if answer.probability is not None else float(answer.value))
        card.said_yes.append(answer.value)
    elif isinstance(answer, ScoreAnswer):
        card.signals.append(answer.score)
        card.said_yes.append(answer.score >= (len(question.levels) - 1) / 2)
    else:
        card.refused += 1
        return
    card.truths.append(label)


def _add(total: Usage, more: Usage) -> Usage:
    return Usage(
        total.input_tokens + more.input_tokens,
        total.output_tokens + more.output_tokens,
        total.cached_tokens + more.cached_tokens,
        total.reasoning_tokens + more.reasoning_tokens,
    )


def evaluate(cases: Iterable[Case], decider: Decider, name: str) -> Scorecard:
    """Ask ``decider`` every case and score its answers."""
    card = Scorecard(name)
    for case in cases:
        start = time.perf_counter()
        decision = decider.decide(case.request)
        card.calls += 1
        card.seconds.append(decision.latency_s or time.perf_counter() - start)
        if decision.usage is not None:
            card.usage_reported += 1
            card.usage = _add(card.usage, decision.usage)
            dollars = cost(decision.usage, name)
            card.cost_usd = None if dollars is None or card.cost_usd is None else card.cost_usd + dollars
        questions = {q.name: q for q in case.request.questions}
        for asked, label in case.labels.items():
            _read(decision.answers.get(asked), questions[asked], label=label, card=card)
    return card


def _fmt(value: float | None, digits: int = 2) -> str:
    return '-' if value is None else f'{value:.{digits}f}'


def _table(cards: Sequence[Scorecard]) -> str:
    head = ('decider', 'calls', 'judged', 'base', 'AUC', 'precision', 'recall', 'choice acc', 'refused',
            'mean s', 'input tok', 'output tok', 'USD')  # fmt: skip
    rows = [
        (
            c.decider,
            str(c.calls),
            str(len(c.truths)),
            _fmt(c.base_rate),
            _fmt(c.auc),
            _fmt(c.precision),
            _fmt(c.recall),
            _fmt(sum(c.choices) / len(c.choices) if c.choices else None),
            str(c.refused),
            _fmt(sum(c.seconds) / len(c.seconds) if c.seconds else None),
            str(c.usage.input_tokens) if c.usage_reported else '-',
            str(c.usage.output_tokens) if c.usage_reported else '-',
            _fmt(c.cost_usd if c.usage_reported else None, 4),
        )
        for c in cards
    ]
    widths = [max(len(r[i]) for r in (head, *rows)) for i in range(len(head))]
    return '\n'.join('  '.join(cell.rjust(w) for cell, w in zip(r, widths, strict=True)) for r in (head, *rows))


def main(argv: Sequence[str] | None = None) -> int:
    """Score every ``--decider`` on the dataset and print one row each."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.evaluate', description=__doc__.splitlines()[0])
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--decider', action='append', required=True, help='a connect() spec; repeatable')
    parser.add_argument('--limit', type=int, default=None, help='only the first N requests')
    parser.add_argument('--ledger', type=Path, default=None, help='serve and record every call through this ledger')
    args = parser.parse_args(argv)
    cases = load(args.dataset, limit=args.limit)
    ledger = None if args.ledger is None else Ledger(args.ledger)
    cards = []
    for spec in args.decider:
        client = connect(spec)
        decider = client if ledger is None else ledger.wrap(client)
        cards.append(evaluate(cases, decider, client.name))
    sys.stdout.write(_table(cards) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
