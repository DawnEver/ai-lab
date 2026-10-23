"""``python -m ai_lab.review run|merge``: findings JSON on stdout, exit 0; a refused invocation exits 2."""

import io
import json
import shutil

import pytest

from ai_lab.review.__main__ import main

pytestmark = pytest.mark.sharp_review  # reads the method data in place

OK = {'schema_version': 1, 'ok': True, 'kind': '', 'result': {'findings': []}, 'usage': None, 'key_slot': 1,
      'error': ''}  # fmt: skip


def _diff(range_, repo):
    assert (range_, repo) == ('main...HEAD', '.')
    return '+y = 2\n'


def test_the_cli_prints_the_findings_json(method_dir, capsys):
    seen = []
    code = main(['run', '--method', str(method_dir), '--range', 'main...HEAD', '--model', 'claude-test'],
                call=lambda r: seen.append(r) or OK, diff=_diff)  # fmt: skip
    assert code == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['schema_version'] == 2
    assert doc['range'] == 'main...HEAD'
    assert {r['status'] for r in doc['reviewers']} == {'ok'}
    assert {r['provider'] for r in seen} == {'anthropic'}


def test_providers_override_the_default(method_dir, capsys):
    main(['run', '--method', str(method_dir), '--range', 'main...HEAD', '--model', 'm', '--providers', 'ghost'],
         call=lambda r: OK, diff=_diff)  # fmt: skip
    doc = json.loads(capsys.readouterr().out)
    assert {(r['provider'], r['status']) for r in doc['reviewers']} == {('ghost', 'unavailable')}


def test_an_unknown_method_version_exits_2_naming_it(method_dir, tmp_path, capsys):
    copy = tmp_path / 'method'
    shutil.copytree(method_dir, copy)
    (copy / 'method.json').write_text('{"schema_version": 99}', encoding='utf-8')
    assert main(['run', '--method', str(copy), '--range', 'a..b', '--model', 'm'], call=None, diff=_diff) == 2
    out = capsys.readouterr()
    assert out.out == ''
    assert 'schema_version 99' in out.err


def test_a_range_git_cannot_diff_exits_2(method_dir, tmp_path, capsys):
    code = main(['run', '--method', str(method_dir), '--range', 'nope..nada', '--model', 'm', '--repo', str(tmp_path)])
    assert code == 2
    out = capsys.readouterr()
    assert out.out == ''
    assert 'nope..nada' in out.err


RUN = {'range': 'HEAD', 'date': '2026-10-20', 'profiles': ['diff'],
       'reviewers': [{'name': 'diff', 'provider': 'claude', 'status': 'pending'}],
       'results': [{'name': 'diff', 'findings': [{'severity': 'HIGH', 'summary': 'S', 'category': 'Bug'}]}],
}  # fmt: skip


def _merge(method_dir, record, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO(record if isinstance(record, str) else json.dumps(record)))
    return main(['merge', '--method', str(method_dir)])


def test_merge_prints_the_findings_json_of_a_run_record_on_stdin(method_dir, monkeypatch, capsys):
    assert _merge(method_dir, RUN, monkeypatch) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['schema_version'] == 2
    assert [f['id'] for f in doc['findings']] == ['SR-20261020-001']
    assert doc['reviewers'][0]['status'] == 'ok'


def test_merge_reads_a_record_without_results_as_every_result_missing(method_dir, monkeypatch, capsys):
    record = {k: v for k, v in RUN.items() if k != 'results'}
    assert _merge(method_dir, record, monkeypatch) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['findings'] == []
    assert doc['reviewers'][0]['status'] != 'ok'


@pytest.mark.parametrize('record', ['not json', {'range': 'HEAD'}, '[]'])
def test_merge_refuses_a_malformed_record_with_exit_2(method_dir, monkeypatch, capsys, record):
    assert _merge(method_dir, record, monkeypatch) == 2
    out = capsys.readouterr()
    assert out.out == ''
    assert out.err.startswith('refused: ')
