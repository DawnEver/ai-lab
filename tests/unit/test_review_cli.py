"""``python -m ai_lab.review`` and ``python -m ai_lab.review.merge``: findings JSON on stdout; refused exits 2."""

import io
import json
import shutil
import subprocess

import pytest

from ai_lab.review import merge as merge_cli
from ai_lab.review.__main__ import _git_diff, main

OK = {'schema_version': 1, 'ok': True, 'kind': '', 'result': {'findings': []}, 'usage': None, 'key_slot': 1,
      'error': ''}  # fmt: skip


def _diff(range_, repo, paths=()):
    assert (range_, repo, tuple(paths)) == ('main...HEAD', '.', ())
    return '+y = 2\n'


def test_the_cli_prints_the_findings_json(method_dir, capsys):
    seen = []
    code = main(['--method', str(method_dir), '--range', 'main...HEAD', '--model', 'claude-test'],
                call=lambda r: seen.append(r) or OK, diff=_diff)  # fmt: skip
    assert code == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['schema_version'] == 2
    assert doc['range'] == 'main...HEAD'
    assert {r['status'] for r in doc['reviewers']} == {'ok'}
    assert {r['provider'] for r in seen} == {'anthropic'}


def test_providers_override_the_default(method_dir, capsys):
    main(['--method', str(method_dir), '--range', 'main...HEAD', '--model', 'm', '--providers', 'ghost'],
         call=lambda r: OK, diff=_diff)  # fmt: skip
    doc = json.loads(capsys.readouterr().out)
    assert {(r['provider'], r['status']) for r in doc['reviewers']} == {('ghost', 'unavailable')}


def test_an_unknown_method_version_exits_2_naming_it(method_dir, tmp_path, capsys):
    copy = tmp_path / 'method'
    shutil.copytree(method_dir, copy)
    (copy / 'method.json').write_text('{"schema_version": 99}', encoding='utf-8')
    assert main(['--method', str(copy), '--range', 'a..b', '--model', 'm'], call=None, diff=_diff) == 2
    out = capsys.readouterr()
    assert out.out == ''
    assert 'schema_version 99' in out.err


def test_a_range_git_cannot_diff_exits_2(method_dir, tmp_path, capsys):
    code = main(['--method', str(method_dir), '--range', 'nope..nada', '--model', 'm', '--repo', str(tmp_path)])
    assert code == 2
    out = capsys.readouterr()
    assert out.out == ''
    assert 'nope..nada' in out.err


def test_the_run_takes_the_verb_less_argv_the_forge_declares(method_dir, capsys):
    # The argv shape `[tool.lab_commons.review] command` declares in every consumer pyproject: no verb.
    argv = ['--method', str(method_dir), '--range', 'main...HEAD', '--model', 'claude-test']
    assert main(argv, call=lambda r: OK, diff=_diff) == 0
    assert json.loads(capsys.readouterr().out)['schema_version'] == 2


def test_the_method_defaults_to_the_packaged_one(capsys):
    assert main(['--range', 'main...HEAD', '--model', 'm'], call=lambda r: OK, diff=_diff) == 0
    assert json.loads(capsys.readouterr().out)['schema_version'] == 2


RUN = {'range': 'HEAD', 'date': '2026-10-20', 'profiles': ['diff'],
       'reviewers': [{'name': 'diff', 'provider': 'claude', 'status': 'pending'}],
       'results': [{'name': 'diff', 'findings': [{'severity': 'HIGH', 'summary': 'S', 'category': 'Bug'}]}],
}  # fmt: skip


def _merge(method_dir, record, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO(record if isinstance(record, str) else json.dumps(record)))
    return merge_cli.main(['--method', str(method_dir)])


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


def test_merge_defaults_to_the_packaged_method(monkeypatch, capsys):
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(RUN)))
    assert merge_cli.main([]) == 0
    assert json.loads(capsys.readouterr().out)['findings'][0]['id'] == 'SR-20261020-001'


@pytest.mark.parametrize('record', ['not json', {'range': 'HEAD'}, '[]'])
def test_merge_refuses_a_malformed_record_with_exit_2(method_dir, monkeypatch, capsys, record):
    assert _merge(method_dir, record, monkeypatch) == 2
    out = capsys.readouterr()
    assert out.out == ''
    assert out.err.startswith('refused: ')


def test_paths_restrict_the_diff_to_the_slice(method_dir, capsys):
    seen = []

    def diff(range_, repo, paths=()):
        seen.append(tuple(paths))
        return '+y = 2\n'

    argv = [
        '--method',
        str(method_dir),
        '--range',
        'a..b',
        '--model',
        'm',
        '--lens',
        'light',
        '--paths',
        'src/x',
        'docs',
    ]
    assert main(argv, call=lambda r: OK, diff=diff) == 0
    assert seen == [('src/x', 'docs')]
    assert len(json.loads(capsys.readouterr().out)['reviewers']) == 1


def test_the_real_diff_reads_only_the_named_paths(tmp_path):

    git = shutil.which('git')

    def g(*a):
        subprocess.run([git, '-C', str(tmp_path), *a], check=True, capture_output=True)

    g('init', '-q')
    g('config', 'user.email', 't@t')
    g('config', 'user.name', 't')
    (tmp_path / 'a.txt').write_text('a\n', encoding='utf-8')
    (tmp_path / 'b.txt').write_text('b\n', encoding='utf-8')
    g('add', '.')
    g('commit', '-qm', 'one')
    (tmp_path / 'a.txt').write_text('a2\n', encoding='utf-8')
    (tmp_path / 'b.txt').write_text('b2\n', encoding='utf-8')
    g('commit', '-qam', 'two')
    text = _git_diff('HEAD~1..HEAD', str(tmp_path), ('a.txt',))
    assert 'a2' in text
    assert 'b2' not in text


def test_synthesis_reads_the_merged_findings_and_the_range_shape(method_dir, tmp_path, capsys):
    merged = tmp_path / 'merged.json'
    merged.write_text(json.dumps({'findings': [{'summary': 'SLICE-FINDING'}]}), encoding='utf-8')
    seen = []

    def diff(range_, repo, paths=()):
        return 'FULL-DIFF-NEVER-READ'

    argv = ['--method', str(method_dir), '--range', 'a..b', '--model', 'm', '--findings', str(merged)]
    code = main(argv, call=lambda r: seen.append(r) or OK, diff=diff, stat=lambda range_, repo: ' x.py | 2 +-\n')
    assert code == 0
    text = seen[0]['context'][0]['text']
    assert 'SLICE-FINDING' in text
    assert 'x.py | 2 +-' in text
    assert 'FULL-DIFF-NEVER-READ' not in text
    assert json.loads(capsys.readouterr().out)['schema_version'] == 2


def test_an_unreadable_findings_file_exits_2(method_dir, tmp_path, capsys):
    argv = ['--method', str(method_dir), '--range', 'a..b', '--model', 'm', '--findings', str(tmp_path / 'no.json')]
    assert main(argv, call=None, diff=_diff) == 2
    assert capsys.readouterr().out == ''
