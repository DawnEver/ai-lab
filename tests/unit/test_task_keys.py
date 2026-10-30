"""A task's own API key: username ``api-key-<task>`` under the provider's host, never served the default.

User ruling 2026-10-19: each task gets its own AI API key, in the one credential scheme
(``lab_commons.credentials``). Every secret here is a fake.
"""

import io

import pytest
from lab_commons.credentials import api_key

from ai_lab import ProviderError, Unsupported, auth, connect
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


def test_a_task_takes_slots_and_task_x_s_2_stores_api_key_x_2(credential_store, monkeypatch, capsys):
    """User ruling 2026-10-19: every task takes several keys, through the one slot argument."""
    _stdin(monkeypatch, 'sk-ant-review-2\n')
    assert auth.main(['login', 'anthropic', '--task', 'pr-review', '-s', '2']) == 0
    assert api_key('api.anthropic.com', task='pr-review', slot=2) == 'sk-ant-review-2'
    assert 'api-key-pr-review-2' in credential_store.file.read_text(encoding='utf-8')


@pytest.mark.parametrize('flag', ['--slot', '--seat', '-s'])
def test_slot_seat_and_s_are_one_argument(credential_store, monkeypatch, flag):
    _stdin(monkeypatch, 'sk-fake-3\n')
    assert auth.main(['login', 'openai', flag, '3']) == 0
    assert api_key('api.openai.com', slot=3) == 'sk-fake-3'


def test_status_lists_the_stored_task_keys(credential_store, only_provider, capsys):
    only_provider('anthropic')
    credential_store('api.anthropic.com', 'sk-default')
    credential_store('api.anthropic.com', 'sk-review', 'api-key-pr-review')
    credential_store('api.anthropic.com', 'sk-review-2', 'api-key-pr-review-2')
    assert auth.main(['status']) == 0
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if x.startswith('anthropic '))
    assert line.split()[-2:] == ['1', 'pr-review:2']
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


def test_a_tasks_slot_2_serves_when_its_slot_1_fails(credential_store, request_):
    credential_store('api.openai.com', 'sk-opt-1', 'api-key-optimization')
    credential_store('api.openai.com', 'sk-opt-2', 'api-key-optimization-2')
    post = _post(ProviderError('HTTP 429 from u: insufficient_quota', kind='exhausted'), DECISIONS_RAW)
    d = connect('openai:gpt-6-luna', post=post, task='optimization').decide(request_)
    assert [c.headers['Authorization'] for c in post.calls] == ['Bearer sk-opt-1', 'Bearer sk-opt-2']
    assert d.key_slot == 2


def test_a_task_never_falls_over_to_another_tasks_or_the_defaults_slots(credential_store, request_):
    """PLANTED control: the default and pr-review each hold a working slot 2; optimization holds slot 1 only."""
    for username, secret in [('api-key', 'sk-d1'), ('api-key-2', 'sk-d2'), ('api-key-pr-review', 'sk-r1'),
                             ('api-key-pr-review-2', 'sk-r2'), ('api-key-optimization', 'sk-o1')]:  # fmt: skip
        credential_store('api.openai.com', secret, username)
    post = _post(ProviderError('x', kind='auth'), DECISIONS_RAW)
    with pytest.raises(ProviderError, match='slot 1: auth'):
        connect('openai:gpt-6-luna', post=post, task='optimization').decide(request_)
    assert [c.headers['Authorization'] for c in post.calls] == ['Bearer sk-o1']
    served = connect('openai:gpt-6-luna', post=_post(ProviderError('x', kind='auth'), DECISIONS_RAW)).decide(request_)
    assert served.key_slot == 2
