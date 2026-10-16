import json

import pytest

from ai_lab import Basis, Provider, ResponseRequest, Text, Unsupported, connect, providers
from ai_lab.client import Client
from ai_lab.transport import CliResult, HttpCall


def _post_returning(payload):
    calls = []

    def post(call, **_):
        calls.append(call)
        return payload

    post.calls = calls
    return post


def test_every_packaged_provider_names_a_shipped_wire():
    assert {'openai', 'typesafe', 'anthropic', 'gemini', 'claude_code', 'codex'} <= set(providers())


def test_a_native_endpoint_decides_natively(keys, request_):
    post = _post_returning({'answers': [
        {'type': 'predicate', 'name': 'worth_fea', 'probability': 0.9},
        {'type': 'choice', 'name': 'fidelity', 'choice': 'fea3d'},
        {'type': 'score', 'name': 'risk', 'score': 2.0},
    ]})  # fmt: skip
    d = connect('openai:gpt-6-luna', post=post).decide(request_)
    assert d.basis is Basis.NATIVE
    assert d['fidelity'].choice == 'fea3d'
    assert d['risk'].level == 'high'
    assert post.calls[0].url.endswith('/decisions')
    assert post.calls[0].headers['Authorization'] == 'Bearer sk-test'


def test_a_responding_wire_decides_by_emulation(keys, request_):
    answer = {'worth_fea': False, 'fidelity': 'drop', 'risk': 'low'}
    post = _post_returning({'content': [{'type': 'text', 'text': json.dumps(answer)}]})
    d = connect('anthropic:claude-x', post=post).decide(request_)
    assert d.basis is Basis.STATED
    assert d['worth_fea'].probability is None
    assert post.calls[0].body['output_config']['format']['schema']['required'] == ['worth_fea', 'fidelity', 'risk']


def test_a_decision_endpoint_cannot_respond(keys):
    with pytest.raises(Unsupported, match='decision endpoint'):
        connect('openai:gpt-6-luna').respond(ResponseRequest('i', (Text('t'),), {'type': 'object'}))


def test_an_image_to_a_blind_provider_is_refused(keys, image_request):
    with pytest.raises(Unsupported, match='does not read images'):
        connect('typesafe:jev-latest').decide(image_request)


def test_a_missing_key_names_the_login_command(credential_store, request_):
    with pytest.raises(Unsupported, match=r'python -m ai_lab\.auth login openai'):
        connect('openai:gpt-6-luna').decide(request_)


def test_an_environment_variable_is_never_read_for_a_key(credential_store, monkeypatch, request_):
    """PLANTED CONTROL: the conventional variable is set and the store is empty -- refused, not read."""
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-from-env')
    with pytest.raises(Unsupported, match='auth login'):
        connect('openai:gpt-6-luna', post=_post_returning({'answers': []})).decide(request_)


def test_the_key_is_read_from_the_store_by_the_endpoint_host(credential_store, request_):
    credential_store('api.openai.com', 'sk-stored')
    post = _post_returning({'answers': [
        {'type': 'predicate', 'name': 'worth_fea', 'probability': 0.9},
        {'type': 'choice', 'name': 'fidelity', 'choice': 'fea3d'},
        {'type': 'score', 'name': 'risk', 'score': 2.0},
    ]})  # fmt: skip
    connect('openai:gpt-6-luna', post=post).decide(request_)
    assert post.calls[0].headers['Authorization'] == 'Bearer sk-stored'


def test_a_keyless_provider_never_asks_the_store(monkeypatch):
    def refuse(host, slot):
        raise AssertionError(host)

    monkeypatch.setattr('ai_lab.client._slot_key', refuse)
    client = Client(providers()['ollama'], 'llama', post=lambda call, **_: dict(call.headers))
    assert client._send(lambda key: HttpCall('http://x', {}, {'key': key})) == ({'key': ''}, 0)


def test_a_user_row_naming_the_retired_key_env_is_refused_with_the_remedy(credential_store, tmp_path):
    (tmp_path / 'home' / 'config').mkdir(parents=True)
    row = '[mylab]\nwire = "openai_chat"\nendpoint = "https://box/v1"\nkey_env = "MYLAB_KEY"\n'
    (tmp_path / 'home' / 'config' / 'providers.toml').write_text(row, encoding='utf-8')
    with pytest.raises(Unsupported, match=r'key_env.*key_name.*ai_lab\.auth login mylab'):
        providers()


def test_an_unknown_provider_lists_the_known_ones():
    with pytest.raises(Unsupported, match='known'):
        connect('nope:x')


def test_an_http_provider_needs_a_model():
    with pytest.raises(Unsupported, match='needs a model'):
        connect('openai')


def test_a_cli_provider_goes_through_run(request_):
    seen = []

    def run(call, **_):
        seen.append(call)
        return CliResult(json.dumps({'worth_fea': True, 'fidelity': 'fea2d', 'risk': 'medium'}))

    d = connect('codex', run=run).decide(request_)
    assert d.basis is Basis.STATED and d['fidelity'].choice == 'fea2d'
    assert seen[0].argv[:2] == ('codex', 'exec')


def test_a_user_table_in_the_family_config_root_adds_and_replaces_rows(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_LAB_HOME', str(tmp_path))
    (tmp_path / 'config').mkdir()
    rows = (
        '[mylab]\nwire = "openai_chat"\nendpoint = "http://box:8000/v1"\nauth = "local"\n'
        '[ollama]\nwire = "openai_chat"\nendpoint = "http://gpu:11434/v1"\nauth = "local"\n'
    )
    (tmp_path / 'config' / 'providers.toml').write_text(rows, encoding='utf-8')
    table = providers()
    assert table['mylab'] == Provider('mylab', 'openai_chat', 'http://box:8000/v1', 'local')
    assert table['ollama'].endpoint == 'http://gpu:11434/v1'


def test_a_row_naming_an_unknown_wire_is_refused():
    with pytest.raises(Unsupported, match='unknown wire'):
        Provider('x', 'smoke_signals', 'e', 'api')


def test_the_client_repr_holds_no_key(keys):
    assert 'sk-test' not in repr(Client(providers()['openai'], 'gpt-x'))


def _user_rows(tmp_path, monkeypatch, rows):
    monkeypatch.setenv('AI_LAB_HOME', str(tmp_path))
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'providers.toml').write_text(rows, encoding='utf-8')


def test_every_packaged_row_declares_its_auth_kind():
    kinds = {name: row.auth for name, row in providers().items()}
    assert kinds['claude_code'] == kinds['codex'] == 'subscription'
    assert {kinds[n] for n in ('anthropic', 'openai')} == {'api'}
    assert kinds['ollama'] == 'local'


def test_a_row_without_auth_is_refused(tmp_path, monkeypatch):
    _user_rows(tmp_path, monkeypatch, '[mylab]\nwire = "openai_chat"\nendpoint = "http://box:8000/v1"\n')
    with pytest.raises(Unsupported, match='missing'):
        providers()


def test_a_row_with_an_unknown_auth_is_refused(tmp_path, monkeypatch):
    _user_rows(tmp_path, monkeypatch, '[mylab]\nwire = "openai_chat"\nendpoint = "e"\nauth = "magic"\n')
    with pytest.raises(Unsupported, match="unknown auth 'magic'"):
        providers()


def _cli_ok(call, **_):
    _cli_ok.calls.append(call)
    return CliResult(
        json.dumps({'type': 'result', 'is_error': False, 'result': '{"ok": true}', 'structured_output': {'ok': True}})
    )


REQ_OK = ResponseRequest('i', (Text('x'),), {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}})


def test_a_subscription_provider_refuses_on_a_slurm_node_and_names_the_api_provider(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '42')
    _cli_ok.calls = []
    with pytest.raises(Unsupported) as refused:
        connect('claude_code', run=_cli_ok).respond(REQ_OK)
    assert not _cli_ok.calls
    text = str(refused.value)
    assert 'claude_code' in text and 'local-only' in text and "'anthropic'" in text
    with pytest.raises(Unsupported, match="'openai'"):
        connect('codex', run=_cli_ok).respond(REQ_OK)


def test_a_subscription_provider_runs_off_the_cluster(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    _cli_ok.calls = []
    connect('claude_code', run=_cli_ok).respond(REQ_OK)
    assert _cli_ok.calls


def test_an_api_provider_runs_on_a_slurm_node(keys, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '42')
    post = _post_returning({'content': [{'type': 'text', 'text': '{"ok": true}'}]})
    connect('anthropic:m', post=post).respond(REQ_OK)
    assert post.calls
