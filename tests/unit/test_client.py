import json

import pytest

from ai_lab import Basis, Provider, ResponseRequest, Text, Unsupported, connect, providers
from ai_lab.client import Client
from ai_lab.transport import CliResult


def _post_returning(payload):
    calls = []

    def post(call, **_):
        calls.append(call)
        return payload

    post.calls = calls
    return post


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'ak-test')


def test_every_packaged_provider_names_a_shipped_wire():
    assert {'openai', 'openai_decisions', 'typesafe', 'anthropic', 'gemini', 'claude_code', 'codex'} <= set(providers())


def test_a_native_endpoint_decides_natively(keys, request_):
    post = _post_returning({'answers': [
        {'type': 'predicate', 'name': 'worth_fea', 'probability': 0.9},
        {'type': 'choice', 'name': 'fidelity', 'choice': 'fea3d'},
        {'type': 'score', 'name': 'risk', 'score': 2.0},
    ]})  # fmt: skip
    d = connect('openai_decisions:gpt-6-luna', post=post).decide(request_)
    assert d.basis is Basis.NATIVE
    assert d['fidelity'].choice == 'fea3d'
    assert d['risk'].level == 'high'
    assert post.calls[0].url.endswith('/decisions')
    assert post.calls[0].headers['Authorization'] == 'Bearer sk-test'


def test_a_responding_wire_decides_by_emulation(keys, request_):
    answer = {'worth_fea': False, 'fidelity': 'drop', 'risk': 'low'}
    post = _post_returning({'content': [{'type': 'tool_use', 'input': answer}]})
    d = connect('anthropic:claude-x', post=post).decide(request_)
    assert d.basis is Basis.STATED
    assert d['worth_fea'].probability is None
    assert post.calls[0].body['tools'][0]['input_schema']['required'] == ['worth_fea', 'fidelity', 'risk']


def test_a_decision_endpoint_cannot_respond(keys):
    with pytest.raises(Unsupported, match='decision endpoint'):
        connect('openai_decisions:gpt-6-luna').respond(ResponseRequest('i', (Text('t'),), {'type': 'object'}))


def test_an_image_to_a_blind_provider_is_refused(monkeypatch, image_request):
    monkeypatch.setenv('TYPESAFE_API_KEY', 'k')
    with pytest.raises(Unsupported, match='does not read images'):
        connect('typesafe:jev-latest').decide(image_request)


def test_a_missing_key_names_its_variable(monkeypatch, request_):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(Unsupported, match='OPENAI_API_KEY'):
        connect('openai_decisions:gpt-6-luna').decide(request_)


def test_an_unknown_provider_lists_the_known_ones():
    with pytest.raises(Unsupported, match='known'):
        connect('nope:x')


def test_an_http_provider_needs_a_model():
    with pytest.raises(ValueError, match='needs a model'):
        connect('openai')


def test_a_cli_provider_goes_through_run(request_):
    seen = []

    def run(call, **_):
        seen.append(call)
        return CliResult(json.dumps({'worth_fea': True, 'fidelity': 'fea2d', 'risk': 'medium'}))

    d = connect('codex', run=run).decide(request_)
    assert d.basis is Basis.STATED and d['fidelity'].choice == 'fea2d'
    assert seen[0].argv[:2] == ('codex', 'exec')


def test_a_user_table_adds_and_replaces_rows(tmp_path, monkeypatch):
    extra = tmp_path / 'p.toml'
    extra.write_text('[mylab]\nwire = "openai_chat"\nendpoint = "http://box:8000/v1"\n', encoding='utf-8')
    monkeypatch.setenv('AI_LAB_PROVIDERS', str(extra))
    assert providers()['mylab'] == Provider('mylab', 'openai_chat', 'http://box:8000/v1')


def test_a_row_naming_an_unknown_wire_is_refused():
    with pytest.raises(ValueError, match='unknown wire'):
        Provider('x', 'smoke_signals', 'e')


def test_the_client_repr_holds_no_key(keys):
    assert 'sk-test' not in repr(Client(providers()['openai'], 'gpt-x'))
