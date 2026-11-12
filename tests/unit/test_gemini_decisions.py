"""Google's Decisions API (``gemini-decision-1``) is a wire of the ``gemini`` account, held to its daily call quota.

Shapes are Google's documented ones (aistudio.google.com/docs/decisions-api, read 2026-11-12); every secret is fake.
"""

import pytest

from ai_lab import CallCapReached, ChoiceAnswer, PredicateAnswer, catalogue, connect
from ai_lab.client import user_providers_file
from ai_lab.wire import gemini_decisions

RAW = {
    'answers': {
        'worth_fea': {'type': 'noul', 'noul': 0.82, 'confidence': 0.7},
        'fidelity': {
            'type': 'choice',
            'choice': 'fea2d',
            'probabilities': {'fea2d': 0.9, 'drop': 0.06, 'fea3d': 0.04},
            'confidence': 0.88,
        },
        'risk': {
            'type': 'score',
            'score': 1.12,
            'legend': {'0': 'low', '1': 'medium', '2': 'high: magnet near demag'},
            'probabilities': {'0': 0.03, '1': 0.83, '2': 0.14},
            'confidence': 0.6,
        },
    },
    'usage': {
        'total_input_tokens': 412,
        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 412}],
        'total_output_tokens': 0,
        'total_tokens': 412,
    },
}


def test_the_request_is_googles_decisions_shape(request_):
    call = gemini_decisions.WIRE.decide(
        request_, 'gemini-decision-1', 'https://generativelanguage.googleapis.com/v1beta', 'gk-x'
    )
    assert call.url == 'https://generativelanguage.googleapis.com/v1beta/decisions'
    assert call.headers == {'x-goog-api-key': 'gk-x'}
    body = call.body
    assert body['model'] == 'gemini-decision-1'
    assert isinstance(body['input'], str) and '"torque_nm": 327.0' in body['input']
    qs = body['questions']
    assert qs['worth_fea'] == {'type': 'noul', 'instructions': 'Is this design worth an expensive FEA?'}
    assert qs['fidelity']['criteria'] == {'drop': 'discard', 'fea2d': 'fea2d', 'fea3d': 'fea3d'}
    assert qs['risk']['criteria'] == ['low', 'medium', 'high: magnet near demag']


def test_an_image_travels_as_a_content_block(image_request):
    blocks = gemini_decisions.WIRE.decide(image_request, 'm', 'e', 'k').body['input']
    assert blocks[0] == {'type': 'text', 'text': 'flux plot'}
    assert blocks[1]['type'] == 'image' and blocks[1]['mime_type'] == 'image/png' and blocks[1]['data']


def test_the_answers_and_the_input_only_usage_are_read(request_):
    got = gemini_decisions.WIRE.parse_decision(RAW, request_)
    assert got['worth_fea'] == PredicateAnswer('worth_fea', True, 0.82)
    assert got['fidelity'] == ChoiceAnswer('fidelity', 'fea2d', {'fea2d': 0.9, 'drop': 0.06, 'fea3d': 0.04}, 0.88)
    assert got['risk'].level == 'medium' and got['risk'].probabilities['high'] == 0.14
    usage = gemini_decisions.WIRE.parse_usage(RAW)
    assert (usage.input_tokens, usage.output_tokens) == (412, 0)


def test_the_decision_model_routes_to_the_decisions_wire_and_the_others_do_not(credential_store, request_):
    credential_store('generativelanguage.googleapis.com', 'gk-test')
    calls = []

    def post(call, **_):
        calls.append(call)
        return RAW

    decision = connect('gemini:gemini-decision-1', post=post).decide(request_)
    assert calls[0].url.endswith('/v1beta/decisions')
    assert decision.answers['worth_fea'] == PredicateAnswer('worth_fea', True, 0.82)
    assert connect('gemini:gemini-3.6-flash').wire.name == 'gemini'


def test_the_packaged_gemini_row_declares_the_free_tier_quota():
    assert catalogue()['gemini'].daily_calls == 10_000


def test_a_call_past_the_declared_daily_quota_is_refused_unsent(credential_store, request_):
    credential_store('generativelanguage.googleapis.com', 'gk-test')
    user = user_providers_file()
    user.parent.mkdir(parents=True, exist_ok=True)
    row = catalogue()['gemini']
    user.write_text(
        '[gemini]\nwire = "gemini"\nurl = "https://generativelanguage.googleapis.com"\ngemini_path = "/v1beta"\n'
        f'key_name = "{row.key_name}"\nvendor = "google"\nauth = "api"\nvision = true\n'
        'model_wires = { "gemini-decision-1" = "gemini_decisions" }\ndaily_calls = 2\n',
        encoding='utf-8',
    )
    calls = []

    def post(call, **_):
        calls.append(call)
        return RAW

    client = connect('gemini:gemini-decision-1', post=post)
    client.decide(request_)
    client.decide(request_)
    with pytest.raises(CallCapReached, match=r'gemini has made 2 of its 2 calls today') as refused:
        client.decide(request_)
    assert refused.value.scope == 'gemini'
    assert len(calls) == 2


def test_a_negative_daily_quota_is_refused():
    from ai_lab import Provider, Unsupported  # noqa: PLC0415

    with pytest.raises(Unsupported, match='daily_calls'):
        Provider(name='x', wire='gemini', url='https://h', auth='api', gemini_path='/v1', daily_calls=-1)
