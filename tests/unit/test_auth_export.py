"""``auth export`` -> ``login --from-file -``: every stored key moves, each under its same username."""

from __future__ import annotations

import io

import pytest
from lab_commons.credentials import key_username, store_credential

from ai_lab import auth

#: (host, task, slot, FAKE key) -- every task, the default, and second slots.
_STORED = [
    ('api.openai.com', None, 1, 'sk-fake-default'),
    ('api.openai.com', None, 2, 'sk-fake-default-2'),
    ('api.openai.com', 'pr-review', 1, 'sk-fake-review'),
    ('api.openai.com', 'pr-review', 2, 'sk-fake-review-2'),
    ('api.anthropic.com', 'optimization', 1, 'sk-fake-opt'),
    ('api.deepseek.com', 'session', 1, 'sk-fake-session'),
]
_HOSTS = sorted({host for host, *_ in _STORED})


class _Pipe(io.StringIO):
    def isatty(self):
        return False


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def _held():
    return {host: auth._held(host) for host in _HOSTS}


def _export(monkeypatch):
    out = _Pipe()
    with monkeypatch.context() as scoped:  # scoped: undo() would also drop the fake credential store
        scoped.setattr('sys.stdout', out)
        assert auth.main(['export']) == 0
    return out.getvalue()


def test_export_then_login_from_file_restores_every_key_under_its_username(credential_store, monkeypatch, tmp_path):
    for host, task, slot, key in _STORED:
        store_credential(host, key_username(task, slot), key)
    before = _held()
    assert before['api.openai.com'] == {
        'api-key': 'sk-fake-default', 'api-key-2': 'sk-fake-default-2',
        'api-key-pr-review': 'sk-fake-review', 'api-key-pr-review-2': 'sk-fake-review-2',
    }  # fmt: skip
    text = _export(monkeypatch)
    credential_store.file.unlink()  # the target machine: an empty store
    assert all(not keys for keys in _held().values())
    monkeypatch.setattr('sys.stdin', io.StringIO(text))
    assert auth.main(['login', '--from-file', '-']) == 0
    assert _held() == before


def test_export_refuses_a_terminal_and_prints_no_key(credential_store, monkeypatch, capsys):
    store_credential('api.openai.com', 'api-key', 'sk-fake-secret')
    with monkeypatch.context() as scoped:
        scoped.setattr('sys.stdout', _Terminal())
        assert auth.main(['export']) == 1
    err = capsys.readouterr().err
    assert 'terminal' in err and 'sk-fake-secret' not in err


@pytest.mark.parametrize(
    ('line', 'parsed'),
    [
        ('OPENAI_API_KEY sk-a', ('OPENAI_API_KEY', None, None, 'sk-a')),
        ('OPENAI_API_KEY=sk-a', ('OPENAI_API_KEY', None, None, 'sk-a')),
        ('OPENAI_API_KEY:2 sk-a', ('OPENAI_API_KEY', None, 2, 'sk-a')),
        ('OPENAI_API_KEY:pr-review sk-a', ('OPENAI_API_KEY', 'pr-review', None, 'sk-a')),
        ('OPENAI_API_KEY:pr-review:2 sk-a', ('OPENAI_API_KEY', 'pr-review', 2, 'sk-a')),
    ],
)
def test_the_parser_reads_task_and_slot(line, parsed):
    assert auth.parse_key_file(line) == ([parsed], [])


def test_writer_and_parser_round_trip():
    lines = [('OPENAI_API_KEY', None, 1, 'a'), ('OPENAI_API_KEY', 'session', 3, 'b')]
    assert auth.parse_key_file(auth.format_key_file(lines))[0] == lines


def test_an_old_name_value_line_lands_in_the_default_slot_1(credential_store, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO('OPENAI_API_KEY sk-fake-old\n'))
    assert auth.main(['login', '--from-file', '-']) == 0
    assert auth._held('api.openai.com') == {'api-key': 'sk-fake-old'}


def test_a_line_task_overrides_no_flag_and_its_slot_defaults_to_1(credential_store, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO('OPENAI_API_KEY:session sk-fake-s\n'))
    assert auth.main(['login', '--from-file', '-']) == 0
    assert auth._held('api.openai.com') == {'api-key-session': 'sk-fake-s'}
