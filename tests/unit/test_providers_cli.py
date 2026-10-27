"""``python -m ai_lab.providers --json`` publishes the resolved catalogue with what proves a copy current."""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from ai_lab import catalogue, providers
from ai_lab.client import packaged_providers_file, user_providers_file
from ai_lab.providers import CATALOGUE_SCHEMA_VERSION, document, main


def test_the_document_is_versioned_and_names_its_inputs_and_producer(credential_store):
    doc = document(now=datetime(2026, 10, 20, tzinfo=UTC))
    assert doc['schema_version'] == CATALOGUE_SCHEMA_VERSION == 1
    assert doc['generated_at'] == '2026-10-20T00:00:00+00:00'
    try:
        installed = version('ai_lab')
    except PackageNotFoundError:
        installed = None
    assert doc['ai_lab_version'] == installed
    assert doc['python'] == sys.executable
    packaged, user = doc['inputs']
    assert packaged['path'] == str(packaged_providers_file()) and packaged['exists']
    assert packaged['sha256'] == hashlib.sha256(packaged_providers_file().read_bytes()).hexdigest()
    assert user == {'path': str(user_providers_file()), 'exists': False, 'sha256': None}
    assert doc['source_sha256'] == packaged['sha256']  # the user file is absent: it adds no bytes
    assert set(doc['providers']) == set(catalogue())


def test_a_source_tree_run_names_no_installed_version(credential_store, monkeypatch):
    # A host that ships `src` on the path (tool.lab_commons.hpc.source_roots) installs no ai_lab dist.
    def absent(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(providers, 'version', absent)
    assert document()['ai_lab_version'] is None


def test_a_user_row_is_resolved_into_the_document_and_moves_the_source_hash(credential_store):
    before = document()['source_sha256']
    user = user_providers_file()
    user.parent.mkdir(parents=True)
    rows = b'[mylab]\nwire = "openai_chat"\nurl = "http://box:8000"\nopenai_path = "/v1"\nauth = "local"\n'
    user.write_bytes(rows)
    doc = document()
    assert doc['providers']['mylab']['endpoints'] == {'openai': 'http://box:8000/v1'}
    assert doc['inputs'][1]['exists'] and doc['source_sha256'] != before
    assert doc['source_sha256'] == hashlib.sha256(packaged_providers_file().read_bytes() + rows).hexdigest()


def test_a_row_carries_its_protocols_role_models_and_key_usernames_never_a_key(credential_store):
    credential_store('api.deepseek.com', 'dk-secret')
    doc = document()
    row = doc['providers']['deepseek']
    assert row['endpoints'] == {
        'anthropic': 'https://api.deepseek.com/anthropic',
        'openai': 'https://api.deepseek.com/v1',
    }
    assert row['protocol'] == 'openai' and row['wire'] == 'openai_chat_json'
    assert row['models']['base'] == 'deepseek-v4-flash[1m]'
    assert row['key_usernames']['default'] == 'api-key'
    assert row['key_usernames']['pr-review'] == 'api-key-pr-review'
    assert doc['providers']['ollama']['key_usernames'] == {}
    assert 'dk-secret' not in json.dumps(doc)


def test_the_cli_prints_the_json_document(credential_store, capsys):
    assert main(['--json']) == 0
    assert json.loads(capsys.readouterr().out)['schema_version'] == CATALOGUE_SCHEMA_VERSION


def test_the_module_runs_as_a_program(credential_store):
    out = subprocess.run(
        [sys.executable, '-m', 'ai_lab.providers', '--json'],
        capture_output=True, text=True, encoding='utf-8', check=True, timeout=120,
    ).stdout  # fmt: skip
    assert 'deepseek' in json.loads(out)['providers']
