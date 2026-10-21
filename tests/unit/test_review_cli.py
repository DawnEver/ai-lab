"""``python -m ai_lab.review``: findings JSON on stdout, exit 0; a refused invocation exits 2 with stdout empty."""

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
