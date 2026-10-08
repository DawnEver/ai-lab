import pytest

from ai_lab import ChoiceAnswer, PredicateAnswer, ProviderError, ScoreAnswer
from ai_lab.emulate import as_response_request, decision_schema, parse_answers


def test_the_schema_is_strict_and_enumerates_every_answer(questions):
    schema = decision_schema(questions)
    assert schema['additionalProperties'] is False
    assert schema['required'] == ['worth_fea', 'fidelity', 'risk']
    assert schema['properties']['worth_fea']['type'] == 'boolean'
    assert schema['properties']['fidelity']['enum'] == ['drop', 'fea2d', 'fea3d']
    assert schema['properties']['risk']['enum'] == ['low', 'medium', 'high']


def test_descriptions_reach_the_model(questions):
    schema = decision_schema(questions)
    assert 'magnet near demag' in schema['properties']['risk']['description']
    assert 'drop: discard' in schema['properties']['fidelity']['description']


def test_stated_answers_carry_no_probability(questions):
    got = parse_answers({'worth_fea': True, 'fidelity': 'fea2d', 'risk': 'high'}, questions)
    assert got['worth_fea'] == PredicateAnswer('worth_fea', True, None)
    assert got['fidelity'] == ChoiceAnswer('fidelity', 'fea2d')
    assert got['risk'] == ScoreAnswer('risk', 'high', 2.0)


@pytest.mark.parametrize(
    ('value', 'match'),
    [
        ({'fidelity': 'fea2d', 'risk': 'low'}, 'no answer'),
        ({'worth_fea': 'yes', 'fidelity': 'fea2d', 'risk': 'low'}, 'true/false'),
        ({'worth_fea': True, 'fidelity': 'cfd', 'risk': 'low'}, 'not one of its options'),
        ({'worth_fea': True, 'fidelity': 'drop', 'risk': 'extreme'}, 'not one of its levels'),
    ],
)
def test_an_answer_outside_the_declared_set_is_refused(questions, value, match):
    with pytest.raises(ProviderError, match=match):
        parse_answers(value, questions)


def test_the_emulated_request_keeps_the_context(request_):
    rr = as_response_request(request_)
    assert rr.context == request_.context
    assert rr.name == 'decision'
