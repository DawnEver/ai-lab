"""``python -m ai_lab.call``: a JSON request in, one classified JSON reply out; a malformed request exits 2."""

import base64
import io
import json
import subprocess
import sys

import pytest

from ai_lab import Ledger, ProviderError
from ai_lab.call import CALL_SCHEMA_VERSION, REPLY_KINDS, Malformed, answer, main
from tests.conftest import PNG

SCHEMA = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}
RESPOND = {
    'verb': 'respond', 'provider': 'anthropic', 'model': 'claude-x', 'instructions': 'Be brief.',
    'context': [{'kind': 'text', 'text': 'hello'}, {'kind': 'fields', 'values': {'n': 1}}], 'schema': SCHEMA,
}  # fmt: skip
DECIDE = {
    'verb': 'decide', 'provider': 'openai', 'model': 'gpt-6-luna', 'instructions': 'Judge the design.',
    'context': [{'kind': 'text', 'text': 'IPM rotor'}],
    'questions': [{'kind': 'predicate', 'name': 'worth_fea', 'instructions': 'Worth an FEA?'}],
}  # fmt: skip
_ANTHROPIC_OK = {
    'content': [{'type': 'text', 'text': '{"ok": true}'}],
    'usage': {'input_tokens': 7, 'output_tokens': 2},
}
_REPLY_KEYS = {'schema_version', 'ok', 'kind', 'result', 'usage', 'key_slot', 'error'}


def _post(payload=None, raises=None):
    calls = []

    def post(call, **_):
        calls.append(call)
        if raises is not None:
            raise raises
        return payload

    post.calls = calls
    return post


def test_respond_returns_the_object_with_its_usage_and_key_slot(keys):
    post = _post(_ANTHROPIC_OK)
    reply = answer(RESPOND, post=post)
    assert set(reply) == _REPLY_KEYS
    assert reply == {
        'schema_version': CALL_SCHEMA_VERSION, 'ok': True, 'kind': '', 'result': {'ok': True},
        'usage': {'input_tokens': 7, 'output_tokens': 2, 'cached_tokens': 0, 'reasoning_tokens': 0},
        'key_slot': 1, 'error': '',
    }  # fmt: skip
    assert post.calls[0].body['system'] == 'Be brief.'


def test_images_ride_after_the_context(keys):
    post = _post(_ANTHROPIC_OK)
    data = base64.b64encode(PNG).decode('ascii')
    answer({**RESPOND, 'images': [{'mime': 'image/png', 'data': data}]}, post=post)
    assert post.calls[0].body['messages'][0]['content'][2]['source']['data'] == data


def test_decide_returns_the_basis_and_typed_answers(keys):
    raw = {'answers': [{'type': 'predicate', 'name': 'worth_fea', 'probability': 0.8}]}
    reply = answer(DECIDE, post=_post(raw))
    assert reply['ok'] and reply['result']['basis'] == 'native'
    assert reply['result']['answers']['worth_fea']['probability'] == 0.8


@pytest.mark.parametrize('kind', sorted(REPLY_KINDS - {'', 'unsupported'}))
def test_a_classified_provider_failure_is_a_reply_not_a_crash(keys, kind):
    reply = answer(RESPOND, post=_post(raises=ProviderError(f'boom {kind}', kind=kind)))
    assert reply['ok'] is False and reply['kind'] == kind and reply['result'] is None
    assert 'boom' in reply['error'] and 'ak-test' not in reply['error']


def test_a_missing_key_or_capability_is_unsupported(credential_store):
    reply = answer(RESPOND, post=_post(_ANTHROPIC_OK))
    assert (reply['ok'], reply['kind']) == (False, 'unsupported') and 'auth login anthropic' in reply['error']
    assert answer({**RESPOND, 'provider': 'nope'})['kind'] == 'unsupported'


def test_the_planted_control_a_success_carries_no_kind(keys):
    assert answer(RESPOND, post=_post(_ANTHROPIC_OK))['kind'] == ''


@pytest.mark.parametrize(
    'request_',
    [
        [],
        {k: v for k, v in RESPOND.items() if k != 'model'},
        {**RESPOND, 'temperature': 0},
        {**RESPOND, 'verb': 'chat'},
        {**RESPOND, 'context': 'hello'},
        {**RESPOND, 'context': [{'kind': 'image', 'data': ''}]},
        {k: v for k, v in RESPOND.items() if k != 'schema'},
        {**RESPOND, 'schema': {'type': 'array'}},
        {**DECIDE, 'schema': SCHEMA},
        {**DECIDE, 'questions': [{'kind': 'choice', 'name': 'c', 'instructions': 'i', 'options': []}]},
        {**RESPOND, 'images': [{'data': 'not base64!'}]},
        {**RESPOND, 'task': 'no-such-task'},
    ],
    ids=['not-object', 'missing', 'unknown', 'verb', 'context-type', 'context-image', 'no-schema', 'bad-schema',
         'decide-schema', 'bad-question', 'bad-image', 'undeclared-task'],
)  # fmt: skip
def test_a_malformed_request_is_refused(keys, request_):
    with pytest.raises(Malformed):
        answer(request_, post=_post(_ANTHROPIC_OK))


def test_main_exits_0_on_a_classified_failure_and_2_on_a_malformed_request(keys, monkeypatch, capsys):
    monkeypatch.setattr('sys.stdin', io.StringIO('{not json'))
    assert main([]) == 2
    out = capsys.readouterr()
    assert out.out == '' and 'malformed' in out.err
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps({**RESPOND, 'provider': 'nope'})))
    assert main([]) == 0
    assert json.loads(capsys.readouterr().out)['kind'] == 'unsupported'


def test_a_configured_ledger_records_the_call_with_its_usage(keys, tmp_path):
    path = tmp_path / 'ledger.jsonl'
    answer(RESPOND, ledger=str(path), post=_post(_ANTHROPIC_OK))
    (entry,) = (json.loads(line) for line in path.read_text('utf-8').splitlines())
    assert entry['usage']['input_tokens'] == 7 and entry['key_slot'] == 1
    assert len(Ledger(path)) == 1


def test_the_module_runs_as_a_program(credential_store):
    done = subprocess.run(
        [sys.executable, '-m', 'ai_lab.call'], input=json.dumps(RESPOND),
        capture_output=True, text=True, encoding='utf-8', check=False, timeout=120,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)['kind'] == 'unsupported'  # no key in the throwaway store
