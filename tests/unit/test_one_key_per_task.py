"""Every task gets its own key: ``login`` refuses a key another credential of the host already holds.

User ruling 2026-10-19. The default keys (``api-key[-N]``) count as their own task. Every secret here is
a fake; ``credential_store`` points git at a throwaway file.
"""

import hashlib
import io

from lab_commons.credentials import api_key, store_credential

from ai_lab import auth

HOST = 'api.anthropic.com'


def _login(monkeypatch, secret, *args):
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{secret}\n'))
    return auth.main(['login', 'anthropic', *args])


def _prefix(secret):
    return hashlib.sha256(secret.encode()).hexdigest()[:12]


def _no_key(text, secret):
    return secret not in text.replace(_prefix(secret), '')


def test_a_key_another_task_holds_is_refused_by_name(credential_store, monkeypatch, capsys):
    store_credential(HOST, 'api-key-pr-review', 'sk-shared')
    assert _login(monkeypatch, 'sk-shared', '--task', 'optimization') == 1
    err = capsys.readouterr().err
    assert 'api-key-pr-review' in err
    assert _prefix('sk-shared') in err
    assert _no_key(err, 'sk-shared')
    assert 'api-key-optimization' not in credential_store.file.read_text(encoding='utf-8')


def test_a_key_the_default_holds_is_refused_for_a_task(credential_store, monkeypatch, capsys):
    store_credential(HOST, 'api-key', 'sk-default')
    assert _login(monkeypatch, 'sk-default', '--task', 'pr-review') == 1
    assert "'api-key'" in capsys.readouterr().err


def test_a_key_another_slot_of_the_same_task_holds_is_refused(credential_store, monkeypatch, capsys):
    store_credential(HOST, 'api-key-pr-review', 'sk-one')
    assert _login(monkeypatch, 'sk-one', '--task', 'pr-review', '-s', '2') == 1
    assert 'api-key-pr-review' in capsys.readouterr().err


def test_restoring_the_same_key_into_the_same_credential_is_a_no_op(credential_store, monkeypatch, capsys):
    store_credential(HOST, 'api-key-pr-review', 'sk-same')
    assert _login(monkeypatch, 'sk-same', '--task', 'pr-review') == 0
    assert api_key(HOST, task='pr-review') == 'sk-same'
    assert 'already' in capsys.readouterr().out


def test_a_distinct_key_still_stores(credential_store, monkeypatch):
    """Planted control: the refusal is about equality, not about a second credential existing."""
    store_credential(HOST, 'api-key-pr-review', 'sk-review')
    assert _login(monkeypatch, 'sk-opt', '--task', 'optimization') == 0
    assert api_key(HOST, task='optimization') == 'sk-opt'


def test_status_lists_a_default_and_task_group_sharing_one_key(credential_store, only_provider, capsys):
    only_provider('anthropic')
    store_credential(HOST, 'api-key', 'sk-migrated')
    store_credential(HOST, 'api-key-pr-review', 'sk-migrated')
    store_credential(HOST, 'api-key-optimization', 'sk-own')
    assert auth.main(['status']) == 0
    out = capsys.readouterr().out
    line = next(row for row in out.splitlines() if _prefix('sk-migrated') in row)
    assert "'api-key'" in line
    assert "'api-key-pr-review'" in line
    assert 'api-key-optimization' not in line
    assert 'delet' in out
    assert _no_key(out, 'sk-migrated')
    assert _prefix('sk-own') not in out
