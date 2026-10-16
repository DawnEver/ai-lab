"""One provider is one account: OpenAI's decisions endpoint is a wire of ``openai``, and an api
provider's keys are an ORDERED list of slots tried in turn on an account-level failure.

Every secret here is a fake.
"""

import pytest

from ai_lab import Provider, ProviderError, ResponseRequest, Text, Unsupported, connect, providers
from ai_lab.client import slot_username, stored_slots

DECISIONS_RAW = {'answers': [
    {'type': 'predicate', 'name': 'worth_fea', 'probability': 0.9},
    {'type': 'choice', 'name': 'fidelity', 'choice': 'fea3d'},
    {'type': 'score', 'name': 'risk', 'score': 2.0},
]}  # fmt: skip
RESPONSES_RAW = {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"a": 1}'}]}]}


def _post(*outcomes):
    """A fake POST answering each call with the next outcome: an exception is raised, else returned."""
    calls = []

    def post(call, **_):
        calls.append(call)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    post.calls = calls
    return post


def test_a_declared_decisions_model_routes_to_the_decisions_wire(keys, request_):
    post = _post(DECISIONS_RAW)
    connect('openai:gpt-6-luna', post=post).decide(request_)
    assert post.calls[0].url == 'https://api.openai.com/v1/decisions'


def test_any_other_openai_model_routes_to_responses(keys):
    post = _post(RESPONSES_RAW)
    got = connect('openai:gpt-x', post=post).respond(ResponseRequest('i', (Text('t'),), {'type': 'object'}))
    assert post.calls[0].url == 'https://api.openai.com/v1/responses'
    assert got.value == {'a': 1}


def test_the_model_wire_table_is_declared_on_the_openai_row():
    assert providers()['openai'].model_wires == {'gpt-6-luna': 'openai_decisions'}


def test_a_model_wire_naming_an_unknown_wire_is_refused():
    with pytest.raises(Unsupported, match='unknown wire'):
        Provider('p', 'openai_responses', 'https://x', 'api', model_wires={'m': 'nope'})


def test_the_retired_decisions_provider_is_refused_naming_its_replacement():
    with pytest.raises(Unsupported, match=r"'openai:gpt-6-luna'"):
        connect('openai_decisions:gpt-6-luna')
    assert 'openai_decisions' not in providers()


def test_slot_usernames():
    assert [slot_username(n) for n in (1, 2, 3)] == ['api-key', 'api-key-2', 'api-key-3']


def test_slot_discovery_stops_at_the_first_gap(credential_store):
    credential_store('api.openai.com', 'sk-1')
    credential_store('api.openai.com', 'sk-2', 'api-key-2')
    credential_store('api.openai.com', 'sk-4', 'api-key-4')
    assert stored_slots('api.openai.com') == 2


def _two_slots(credential_store):
    credential_store('api.openai.com', 'sk-one')
    credential_store('api.openai.com', 'sk-two', 'api-key-2')


@pytest.mark.parametrize(
    'failure',
    [
        ProviderError('HTTP 429 from u: insufficient_quota', kind='exhausted'),
        ProviderError('HTTP 401 from u: bad key', kind='auth'),
        ProviderError('HTTP 429 from u: slow down (after 3 attempts)', kind='rate_limit'),
    ],
    ids=['exhausted', 'auth', 'rate-limit'],
)
def test_an_account_failure_on_slot_1_is_served_by_slot_2(credential_store, request_, failure):
    _two_slots(credential_store)
    post = _post(failure, DECISIONS_RAW)
    d = connect('openai:gpt-6-luna', post=post).decide(request_)
    assert [c.headers['Authorization'] for c in post.calls] == ['Bearer sk-one', 'Bearer sk-two']
    assert d.key_slot == 2


def test_a_non_account_failure_does_not_fail_over(credential_store, request_):
    _two_slots(credential_store)
    post = _post(ProviderError('HTTP 400 from u: bad request'), DECISIONS_RAW)
    with pytest.raises(ProviderError, match='HTTP 400'):
        connect('openai:gpt-6-luna', post=post).decide(request_)
    assert len(post.calls) == 1


def test_all_slots_failing_names_every_kind_and_no_key(credential_store, request_):
    _two_slots(credential_store)
    post = _post(ProviderError('x', kind='exhausted'), ProviderError('y', kind='auth'))
    with pytest.raises(ProviderError) as caught:
        connect('openai:gpt-6-luna', post=post).decide(request_)
    text = str(caught.value)
    assert 'slot 1: exhausted' in text and 'slot 2: auth' in text
    assert 'sk-one' not in text and 'sk-two' not in text


def test_slot_1_serves_when_it_works(keys, request_):
    assert connect('openai:gpt-6-luna', post=_post(DECISIONS_RAW)).decide(request_).key_slot == 1
