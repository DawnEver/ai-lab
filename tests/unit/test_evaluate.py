"""A decider is scored on labelled requests: round-tripped JSON, AUC, precision, recall, choices, ledger."""

from __future__ import annotations

import json

from ai_lab import Choice, DecisionRequest, Fields, Image, Level, Option, Predicate, Rules, Score, Text
from ai_lab.evaluate import Case, auc, evaluate, load
from ai_lab.spec import canonical, decision_request_from_json, decision_request_to_json

REQUEST = DecisionRequest(
    (Text('t'), Fields({'x': 0.2}), Image(b'\x89PNG', 'image/png')),
    (
        Predicate('p0', 'small?'),
        Choice('c0', 'which?', (Option('a', 'first'), Option('b'))),
        Score('s0', 'how good?', (Level('low'), Level('mid'), Level('high'))),
    ),
)


def test_a_request_round_trips_through_json_exactly() -> None:
    data = json.loads(json.dumps(decision_request_to_json(REQUEST)))
    assert canonical(decision_request_from_json(data)) == canonical(REQUEST)


def test_auc_is_half_for_chance_one_for_perfect_and_none_without_both_classes() -> None:
    assert auc([0.9, 0.1], [True, False]) == 1.0
    assert auc([0.5, 0.5], [True, False]) == 0.5
    assert auc([0.9], [True]) is None


def test_a_rule_decider_is_scored_on_every_label_kind() -> None:
    rules = Rules({'p0': lambda f: f['x'] < 0.5, 'c0': lambda f: 'a', 's0': lambda f: 'high'})
    card = evaluate([Case(REQUEST, {'p0': True, 'c0': 'b', 's0': False})], rules, 'rules')
    assert card.truths == [True, False] and card.said_yes == [True, True]
    assert (card.precision, card.recall, card.choices) == (0.5, 1.0, [False])


def test_a_dataset_file_loads_with_a_limit(tmp_path) -> None:
    path = tmp_path / 'd.jsonl'
    line = json.dumps({'request': decision_request_to_json(REQUEST), 'labels': {'p0': True}})
    path.write_text(f'{line}\n{line}\n', 'utf-8')
    assert len(load(path)) == 2
    assert len(load(path, limit=1)) == 1
