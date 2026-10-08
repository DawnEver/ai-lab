"""The two native decision wires, against the request/response shapes their vendors document."""

import pytest

from ai_lab import ChoiceAnswer, DecisionRequest, Fields, PredicateAnswer, ProviderError, Refusal, ScoreAnswer
from ai_lab.wire import openai_decisions, typesafe


def test_openai_decisions_request(request_):
    call = openai_decisions.WIRE.decide(request_, 'gpt-6-luna', 'https://api.openai.com/v1', 'sk-x')
    assert call.url == 'https://api.openai.com/v1/decisions'
    assert call.headers['Authorization'] == 'Bearer sk-x'
    body = call.body
    assert body['model'] == 'gpt-6-luna'
    assert isinstance(body['input'], str) and '"torque_nm": 327.0' in body['input']
    kinds = [(q['type'], q['name']) for q in body['questions']]
    assert kinds == [('predicate', 'worth_fea'), ('choice', 'fidelity'), ('score', 'risk')]
    assert body['questions'][1]['choices'][0] == {'value': 'drop', 'description': 'discard'}
    assert body['questions'][2]['levels'][2] == {'label': 'high', 'description': 'magnet near demag'}


def test_openai_decisions_carries_images_as_data_urls(image_request):
    body = openai_decisions.WIRE.decide(image_request, 'm', 'e', 'k').body
    content = body['input'][0]['content']
    assert content[1]['type'] == 'input_image'
    assert content[1]['image_url'].startswith('data:image/png;base64,')


def test_the_key_never_appears_in_a_call_repr(request_):
    assert 'sk-secret' not in repr(openai_decisions.WIRE.decide(request_, 'm', 'e', 'sk-secret'))


def test_openai_decisions_response(request_):
    raw = {
        'answers': [
            {'type': 'predicate', 'name': 'worth_fea', 'probability': 0.92},
            {
                'type': 'choice',
                'name': 'fidelity',
                'choice': 'fea2d',
                'confidence': 0.9,
                'probabilities': [
                    {'value': 'fea2d', 'probability': 0.9},
                    {'value': 'fea3d', 'probability': 0.08},
                    {'value': 'drop', 'probability': 0.02},
                ],
            },
            {
                'type': 'score',
                'name': 'risk',
                'score': 1.1,
                'confidence': 0.55,
                'probabilities': [
                    {'value': 0, 'label': 'low', 'probability': 0.1},
                    {'value': 1, 'label': 'medium', 'probability': 0.7},
                    {'value': 2, 'label': 'high', 'probability': 0.2},
                ],
            },
        ]
    }
    got = openai_decisions.WIRE.parse_decision(raw, request_)
    assert got['worth_fea'] == PredicateAnswer('worth_fea', True, 0.92)
    assert got['fidelity'].choice == 'fea2d' and got['fidelity'].probabilities['drop'] == 0.02
    assert got['risk'] == ScoreAnswer('risk', 'medium', 1.1, {'low': 0.1, 'medium': 0.7, 'high': 0.2}, 0.55)


def test_openai_decisions_refusal_is_an_answer_not_a_crash(questions):
    req = DecisionRequest((Fields({'a': 1}),), questions[:1])
    got = openai_decisions.WIRE.parse_decision({'answers': [{'type': 'refusal', 'name': 'worth_fea'}]}, req)
    assert got['worth_fea'] == Refusal('worth_fea')


def test_a_missing_answer_is_refused(request_):
    with pytest.raises(ProviderError, match='no answer'):
        openai_decisions.WIRE.parse_decision({'answers': []}, request_)


def test_typesafe_sends_a_lone_fields_part_as_an_object(questions):
    req = DecisionRequest((Fields({'torque_nm': 327.0}),), questions)
    call = typesafe.WIRE.decide(req, 'jev-latest', 'https://api.typesafe.ai/v1', 'k')
    assert call.url == 'https://api.typesafe.ai/v1/systemone'
    assert call.body['state'] == {'torque_nm': 327.0}
    qs = call.body['questions']
    assert qs['worth_fea'] == {'type': 'noul', 'instructions': 'Is this design worth an expensive FEA?'}
    assert qs['fidelity']['criteria'] == {'drop': 'discard', 'fea2d': 'fea2d', 'fea3d': 'fea3d'}
    assert qs['risk']['criteria'] == ['low', 'medium', 'high: magnet near demag']


def test_typesafe_sends_mixed_context_as_text(request_):
    state = typesafe.WIRE.decide(request_, 'm', 'e', 'k').body['state']
    assert isinstance(state, str) and 'IPM rotor' in state


def test_typesafe_response(request_):
    raw = {
        'model': 'jev-1.13.0',
        'answers': {
            'worth_fea': {'type': 'noul', 'noul': 0.3},
            'fidelity': {
                'type': 'choice',
                'choice': 'drop',
                'probabilities': {'drop': 0.8, 'fea2d': 0.2},
                'confidence': 0.81,
            },
            'risk': {'type': 'score', 'score': 0.4, 'probabilities': {'0': 0.6, '1': 0.4, '2': 0.0}, 'confidence': 0.9},
        },
    }
    got = typesafe.WIRE.parse_decision(raw, request_)
    assert got['worth_fea'] == PredicateAnswer('worth_fea', False, 0.3)
    assert got['fidelity'] == ChoiceAnswer('fidelity', 'drop', {'drop': 0.8, 'fea2d': 0.2}, 0.81)
    assert got['risk'].level == 'low' and got['risk'].probabilities == {'low': 0.6, 'medium': 0.4, 'high': 0.0}
