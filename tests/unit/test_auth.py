"""``python -m ai_lab.auth`` -- keys go into a REAL git credential store (a throwaway ``store`` file), never argv.

Every secret here is a fake; ``credential_store`` points git at a file under ``tmp_path``.
"""

import io
import subprocess
import sys

import pytest
from lab_commons.credentials import credential_for

from ai_lab import auth


def _stdin(monkeypatch, text):
    monkeypatch.setattr('sys.stdin', io.StringIO(text))


def test_login_reads_the_key_from_stdin_and_stores_it_under_the_providers_host(credential_store, monkeypatch, capsys):
    _stdin(monkeypatch, 'sk-fake-1\n')
    assert auth.main(['login', 'openai']) == 0
    assert credential_for('api.openai.com') == ('api-key', 'sk-fake-1')
    out = capsys.readouterr()
    assert 'sk-fake-1' not in out.out + out.err


def test_login_slot_2_stores_the_second_username(credential_store, monkeypatch, capsys):
    _stdin(monkeypatch, 'sk-fake-two\n')
    assert auth.main(['login', 'openai', '--slot', '2']) == 0
    assert 'https://api-key-2:sk-fake-two@api.openai.com' in credential_store.file.read_text(encoding='utf-8')
    out = capsys.readouterr()
    assert 'sk-fake-two' not in out.out + out.err


def test_login_refuses_slot_zero(credential_store, monkeypatch, capsys):
    _stdin(monkeypatch, 'sk-x\n')
    assert auth.main(['login', 'openai', '--slot', '0']) == 1
    assert 'slot' in capsys.readouterr().err


def test_login_refuses_an_unknown_or_keyless_provider_and_an_empty_key(credential_store, monkeypatch, capsys):
    assert auth.main(['login', 'nope']) == 1
    assert auth.main(['login', 'ollama']) == 1
    _stdin(monkeypatch, '\n')
    assert auth.main(['login', 'openai']) == 1
    err = capsys.readouterr().err
    assert 'unknown provider' in err
    assert 'needs no key' in err
    assert 'empty' in err


def test_the_key_never_travels_in_argv():
    """``login`` takes a provider and nothing else: a key on the command line lands in shell history."""
    with pytest.raises(SystemExit):
        auth.main(['login', 'openai', 'sk-in-argv'])


@pytest.mark.parametrize(
    'text',
    [
        'OPENAI_API_KEY=sk-o\nDEEPSEEK_API_KEY=sk-d\nTYPESAFE_API_KEY=ts\n',
        'export OPENAI_API_KEY=sk-o\nexport DEEPSEEK_API_KEY="sk-d"\nexport TYPESAFE_API_KEY=\'ts\'\n',
        'OPENAI_API_KEY sk-o\r\nDEEPSEEK_API_KEY\tsk-d\r\nTYPESAFE_API_KEY    ts',
        '# keys\n\n  OPENAI_API_KEY = "sk-o"  \r\n\r\nDEEPSEEK_API_KEY  sk-d\n   # c\nTYPESAFE_API_KEY=ts',
        '﻿OPENAI_API_KEY sk-o\nDEEPSEEK_API_KEY=sk-d\nTYPESAFE_API_KEY ts\n',
    ],
    ids=['equals', 'export-and-quotes', 'whitespace-crlf-no-final-newline', 'comments-blanks-spaces', 'bom'],
)
def test_from_file_accepts_every_line_form(credential_store, tmp_path, capsys, text):
    keys = tmp_path / 'keys'
    keys.write_bytes(text.encode('utf-8'))
    assert auth.main(['login', '--from-file', str(keys)]) == 0
    assert credential_for('api.openai.com')[1] == 'sk-o'
    assert credential_for('api.deepseek.com')[1] == 'sk-d'
    assert credential_for('api.typesafe.ai')[1] == 'ts'
    out = capsys.readouterr()
    assert not {'sk-o', 'sk-d'} & set((out.out + out.err).split())


def test_from_file_reports_an_unknown_name_by_name_never_its_value(credential_store, tmp_path, capsys):
    keys = tmp_path / 'keys'
    keys.write_text('MYSTERY_TOKEN s3cr3t-fake\nOPENAI_API_KEY sk-o\n', encoding='utf-8')
    assert auth.main(['login', '--from-file', str(keys)]) == 0
    out = capsys.readouterr()
    assert 'MYSTERY_TOKEN' in out.err
    assert 's3cr3t-fake' not in out.out + out.err
    assert credential_for('api.openai.com')[1] == 'sk-o'


def test_from_file_dash_reads_stdin(credential_store, monkeypatch):
    _stdin(monkeypatch, 'OPENAI_API_KEY sk-piped\n')
    assert auth.main(['login', '--from-file', '-']) == 0
    assert credential_for('api.openai.com')[1] == 'sk-piped'


def test_a_file_with_nothing_storable_is_a_failure(credential_store, tmp_path):
    keys = tmp_path / 'keys'
    keys.write_text('# nothing\nMYSTERY_TOKEN x\n', encoding='utf-8')
    assert auth.main(['login', '--from-file', str(keys)]) == 1


def test_status_lists_each_provider_with_its_auth_kind_and_key_state_never_the_secret(credential_store, capsys):
    credential_store('api.openai.com', 'sk-fake-status')
    assert auth.main(['status']) == 0
    out = capsys.readouterr().out
    lines = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert lines['openai'].split()[-2:] == ['1', '-']
    assert lines['anthropic'].split()[-2:] == ['0', '-']
    assert 'openai_decisions' not in lines
    assert lines['openai'].split()[1] == 'api'
    assert lines['ollama'].split()[1] == 'local'
    assert 'sk-fake-status' not in out


def test_status_counts_slots_up_to_the_first_gap(credential_store, capsys):
    credential_store('api.openai.com', 'sk-a')
    credential_store('api.openai.com', 'sk-b', 'api-key-2')
    credential_store('api.openai.com', 'sk-d', 'api-key-4')
    assert auth.main(['status']) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith('openai '))
    assert line.split()[-2] == '2'


def test_the_module_runs_as_a_cli():
    done = subprocess.run(
        [sys.executable, '-m', 'ai_lab.auth', '--help'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert 'login' in done.stdout and 'status' in done.stdout


def test_from_file_honours_task_and_slot(credential_store, tmp_path, capsys):
    keys = tmp_path / 'keys'
    keys.write_text('OPENAI_API_KEY sk-task-two\n', encoding='utf-8')
    assert auth.main(['login', '--from-file', str(keys), '--task', 'optimization', '-s', '2']) == 0
    text = credential_store.file.read_text(encoding='utf-8')
    assert 'https://api-key-optimization-2:sk-task-two@api.openai.com' in text
    assert 'https://api-key:' not in text
    out = capsys.readouterr()
    assert 'task optimization slot 2' in out.out
    assert 'sk-task-two' not in out.out + out.err


def test_from_file_refuses_an_unknown_task_before_storing(credential_store, tmp_path, capsys):
    keys = tmp_path / 'keys'
    keys.write_text('OPENAI_API_KEY sk-x\n', encoding='utf-8')
    assert auth.main(['login', '--from-file', str(keys), '--task', 'nope']) == 1
    assert not credential_store.file.exists() or 'sk-x' not in credential_store.file.read_text(encoding='utf-8')


def test_provider_with_from_file_imports_only_that_providers_key(credential_store, tmp_path):
    keys = tmp_path / 'keys'
    keys.write_text('OPENAI_API_KEY sk-o\nDEEPSEEK_API_KEY sk-d\n', encoding='utf-8')
    assert auth.main(['login', 'openai', '--from-file', str(keys), '--task', 'pr-review']) == 0
    text = credential_store.file.read_text(encoding='utf-8')
    assert 'https://api-key-pr-review:sk-o@api.openai.com' in text
    assert 'sk-d' not in text


def test_login_needs_a_provider_or_a_file():
    with pytest.raises(SystemExit):
        auth.main(['login'])
