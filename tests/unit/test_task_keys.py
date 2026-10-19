"""A task's own API key: username ``api-key-<task>`` under the provider's host, never served the default.

User ruling 2026-10-19: each task gets its own AI API key, in the one credential scheme
(``lab_commons.credentials``). Every secret here is a fake.
"""

import io

import pytest
from lab_commons.credentials import api_key

from ai_lab import Unsupported, auth, connect
from ai_lab.client import login_command
from tests.unit.test_one_account import DECISIONS_RAW, _post


def _stdin(monkeypatch, text):
    monkeypatch.setattr('sys.stdin', io.StringIO(text))


def test_login_task_stores_the_task_username_under_the_providers_host(credential_store, monkeypatch, capsys):
    _stdin(monkeypatch, 'sk-ant-review\n')
    assert auth.main(['login', 'anthropic', '--task', 'pr-review']) == 0
    assert api_key('api.anthropic.com', task='pr-review') == 'sk-ant-review'
    out = capsys.readouterr()
    assert 'sk-ant-review' not in out.out + out.err
    assert 'pr-review' in out.out


def test_login_refuses_an_undeclared_task_and_stores_nothing(credential_store, monkeypatch, capsys):
    _stdin(monkeypatch, 'sk-ant-x\n')
    assert auth.main(['login', 'anthropic', '--task', 'pr_reveiw']) == 1
    assert 'pr-review' in capsys.readouterr().err
    assert not credential_store.file.exists() or 'sk-ant-x' not in credential_store.file.read_text(encoding='utf-8')


def test_a_task_key_has_no_slots(credential_store, monkeypatch, capsys):
    _stdin(monkeypatch, 'sk-ant-x\n')
    assert auth.main(['login', 'anthropic', '--task', 'pr-review', '--slot', '2']) == 1
    assert '--task' in capsys.readouterr().err


def test_status_lists_the_stored_task_keys(credential_store, capsys):
    credential_store('api.anthropic.com', 'sk-default')
    credential_store('api.anthropic.com', 'sk-review', 'api-key-pr-review')
    assert auth.main(['status']) == 0
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if x.startswith('anthropic '))
    assert line.split()[-2:] == ['1', 'pr-review']
    assert 'sk-' not in out


def test_a_task_client_sends_the_task_key(credential_store, request_):
    credential_store('api.openai.com', 'sk-default')
    credential_store('api.openai.com', 'sk-opt', 'api-key-optimization')
    post = _post(DECISIONS_RAW)
    connect('openai:gpt-6-luna', post=post, task='optimization').decide(request_)
    assert 'sk-opt' in str(dict(post.calls[0].headers))


def test_a_missing_task_key_is_refused_by_name_with_the_default_stored(credential_store, request_):
    """PLANTED control: the default key is stored and serves the same client without a task."""
    credential_store('api.openai.com', 'sk-default')
    post = _post(DECISIONS_RAW, DECISIONS_RAW)
    connect('openai:gpt-6-luna', post=post).decide(request_)
    with pytest.raises(Unsupported, match='--task optimization'):
        connect('openai:gpt-6-luna', post=post, task='optimization').decide(request_)
    assert len(post.calls) == 1
    assert login_command('openai', 'optimization') == 'python -m ai_lab.auth login openai --task optimization'


def test_an_undeclared_task_is_refused_at_connect():
    with pytest.raises(ValueError, match='pr-review'):
        connect('openai:gpt-6-luna', task='nope')
