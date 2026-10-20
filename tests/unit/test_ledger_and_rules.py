import json

import pytest

from ai_lab import (
    Basis,
    ChoiceAnswer,
    Decider,
    Decision,
    DecisionRequest,
    Fields,
    Ledger,
    PredicateAnswer,
    ReplayMiss,
    Responder,
    Response,
    ResponseRequest,
    Rules,
    ScoreAnswer,
    Text,
    Unsupported,
)
from ai_lab.ledger import LEDGER_SCHEMA_VERSION, UPGRADES
from ai_lab.report import summarize
from ai_lab.usage import Usage


class Counting:
    name = 'fake:m'

    def __init__(self):
        self.calls = 0

    def decide(self, request):
        self.calls += 1
        return Decision(
            {
                'worth_fea': PredicateAnswer('worth_fea', True, 0.9),
                'fidelity': ChoiceAnswer('fidelity', 'fea2d', {'fea2d': 1.0}, 0.9),
                'risk': ScoreAnswer('risk', 'low', 0.2, None, None),
            },
            Basis.NATIVE,
            'fake',
            'm',
            0.1,
        )

    def respond(self, request):
        self.calls += 1
        return Response({'ok': True}, 'fake', 'm', 0.2)


def test_a_recorded_decision_replays_identically_without_the_client(tmp_path, request_):
    path = tmp_path / 'ledger.jsonl'
    client = Counting()
    first = Ledger(path, 'record').wrap(client).decide(request_)
    again = Ledger(path, 'record').wrap(client).decide(request_)
    assert client.calls == 1
    assert again == first
    replayed = Ledger(path, 'replay').wrap(Counting()).decide(request_)
    assert replayed == first


def test_replay_never_calls_a_model_on_a_miss(tmp_path, request_):
    client = Counting()
    with pytest.raises(ReplayMiss, match='record'):
        Ledger(tmp_path / 'l.jsonl', 'replay').wrap(client).decide(request_)
    assert client.calls == 0


def test_a_different_request_is_a_different_entry(tmp_path, request_, questions):
    ledger = Ledger(tmp_path / 'l.jsonl')
    recorded = ledger.wrap(Counting())
    recorded.decide(request_)
    recorded.decide(DecisionRequest((Fields({'torque_nm': 328.0}),), questions))
    assert len(ledger) == 2


def test_responses_are_recorded_too(tmp_path):
    path = tmp_path / 'l.jsonl'
    req = ResponseRequest('i', (Text('t'),), {'type': 'object'})
    client = Counting()
    Ledger(path).wrap(client).respond(req)
    assert Ledger(path, 'replay').wrap(client).respond(req).value == {'ok': True}
    assert client.calls == 1


def test_rules_read_fields_and_answer_by_value(request_):
    rules = Rules(
        {
            'worth_fea': lambda f: f['torque_nm'] > 300,
            'fidelity': lambda f: 'fea2d' if f['ripple_pct'] < 5 else 'drop',
            'risk': lambda f: 'medium',
        }
    )
    d = rules.decide(request_)
    assert d.basis is Basis.RULE
    assert d['worth_fea'] == PredicateAnswer('worth_fea', True)
    assert d['fidelity'].choice == 'fea2d'
    assert d['risk'].score == 1.0


def test_a_question_without_a_rule_is_refused(request_):
    with pytest.raises(Unsupported, match='no rule'):
        Rules({'worth_fea': lambda f: True}).decide(request_)


def test_everything_satisfies_the_two_verbs(tmp_path):
    assert isinstance(Rules({}), Decider)
    recorded = Ledger(tmp_path / 'l.jsonl').wrap(Counting())
    assert isinstance(recorded, Decider) and isinstance(recorded, Responder)


_V1_USAGE = {'input_tokens': 7, 'output_tokens': 2, 'cached_tokens': 0, 'reasoning_tokens': 0, 'cost_usd': 0.5}


def _rewrite(path, **changes):
    entry = json.loads(path.read_text('utf-8'))
    entry.update(changes)
    entry = {k: v for k, v in entry.items() if v is not None}
    path.write_text(json.dumps(entry) + '\n', 'utf-8')


def _v1_entry(path, request_):
    """A schema-1 ledger line: no ``schema_version``, and a usage block carrying vendor dollars."""
    Ledger(path).wrap(Counting()).decide(request_)
    _rewrite(path, schema_version=None, usage=dict(_V1_USAGE))


def test_a_schema_1_entry_is_upgraded_on_read_and_written_back_at_the_current_schema(tmp_path, request_):
    path = tmp_path / 'l.jsonl'
    _v1_entry(path, request_)
    assert Ledger(path, 'replay').wrap(Counting()).decide(request_).usage == Usage(7, 2, 0, 0)
    stored = json.loads(path.read_text('utf-8'))
    assert stored['schema_version'] == LEDGER_SCHEMA_VERSION
    assert 'cost_usd' not in stored['usage']


def test_a_schema_the_reader_does_not_know_is_refused(tmp_path, request_):
    path = tmp_path / 'l.jsonl'
    Ledger(path).wrap(Counting()).decide(request_)
    _rewrite(path, schema_version=99)
    with pytest.raises(Unsupported, match='schema 99'):
        Ledger(path)


def test_a_current_entry_carrying_the_retired_usage_field_is_refused_by_name(tmp_path, request_):
    path = tmp_path / 'l.jsonl'
    Ledger(path).wrap(Counting()).decide(request_)
    _rewrite(path, usage=dict(_V1_USAGE))
    with pytest.raises(Unsupported, match="usage 'cost_usd'"):
        Ledger(path)


def test_the_report_reads_a_schema_1_ledger_through_the_same_upgrade(tmp_path, request_):
    path = tmp_path / 'l.jsonl'
    _v1_entry(path, request_)
    (row,) = summarize([path], price=lambda *_a, **_k: None)
    assert (row.input_tokens, row.cost_usd) == (7, None)


def test_every_older_ledger_schema_declares_its_upgrade_step():
    assert set(UPGRADES) == set(range(1, LEDGER_SCHEMA_VERSION))
