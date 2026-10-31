"""``ai_lab.review.summary`` and disposed ids: the derived "what this change does" section -- network-free."""

import json
import shutil
import subprocess

import pytest

from ai_lab.review.__main__ import main
from ai_lab.review.method import load_method
from ai_lab.review.rules import finding_id, merge
from ai_lab.review.summary import change_summary, git_intent, parse_log
from tests.review_principles import PRINCIPLES, PRINCIPLES_FILE
from tests.unit.test_review_run import _errors

LOG = (
    'aaaaaaa1\x00feat(x): add the thing\n\nWhy: callers needed it.\n\x1e'
    'bbbbbbb2\x00chore(merge): pr/y into integrate/main\n\n'
    'src/a.py: newer-structure -- took upstream helper, re-expressed ours\n'
    'Merge-Audit: LOST tests/t.py::test_old -- superseded by test_new\n'
    'Claude-Session: https://x\n\x1e'
)
INTENT = {
    'commits': [{'sha': 'aaaaaaa1', 'subject': 'feat(x): add the thing', 'body': ''}],
    'memory': [{'path': '.claude/memory/m.md', 'text': 'We add the thing because...'}],
    'resolutions': [], 'merge_audit': [], 'pr_body': '',
}  # fmt: skip


def _reply(result=None, *, ok=True):
    return {'schema_version': 1, 'ok': ok, 'kind': '' if ok else 'timeout', 'result': result, 'usage': None,
            'key_slot': 1, 'error': '' if ok else 'it timed out'}  # fmt: skip


def test_the_log_yields_commits_resolutions_and_merge_audit_lines():
    intent = parse_log(LOG)
    assert [c['sha'] for c in intent['commits']] == ['aaaaaaa1', 'bbbbbbb2']
    assert intent['commits'][0]['subject'] == 'feat(x): add the thing'
    assert intent['resolutions'] == [
        {'sha': 'bbbbbbb2', 'line': 'src/a.py: newer-structure -- took upstream helper, re-expressed ours'}
    ]
    assert intent['merge_audit'] == [{'sha': 'bbbbbbb2', 'line': 'LOST tests/t.py::test_old -- superseded by test_new'}]


def test_the_summary_keeps_only_claims_citing_a_known_source():
    seen = []
    result = {'changes': [{'claim': 'adds the thing', 'cites': ['aaaaaaa1', '.claude/memory/m.md']},
                          {'claim': 'invented', 'cites': ['deadbeef']}],
              'open_items': [{'item': 'docs pending', 'cites': ['.claude/memory/m.md']}]}  # fmt: skip
    got = change_summary(INTENT, call=lambda r: seen.append(r) or _reply(result), provider='anthropic', model='m')
    assert got['changes'] == [{'claim': 'adds the thing', 'cites': ['aaaaaaa1', '.claude/memory/m.md']}]
    assert got['open_items'] == [{'item': 'docs pending', 'cites': ['.claude/memory/m.md']}]
    assert got['sources'] == {'commits': ['aaaaaaa1'], 'memory': ['.claude/memory/m.md']}
    assert 'We add the thing because' in seen[0]['context'][0]['text']
    assert 'error' not in got


def test_a_failed_summary_call_falls_back_to_the_commit_subjects_and_names_the_error():
    got = change_summary(INTENT, call=lambda r: _reply(ok=False), provider='anthropic', model='m')
    assert got['changes'] == [{'claim': 'feat(x): add the thing', 'cites': ['aaaaaaa1']}]
    assert got['error'] == 'timeout: it timed out'


def test_tests_altered_come_from_merge_audit_lines_never_the_model():
    intent = {**parse_log(LOG), 'memory': [], 'pr_body': ''}
    got = change_summary(intent, call=lambda r: _reply({'changes': [], 'open_items': []}), provider='p', model='m')
    assert got['tests_altered'] == [{'line': 'LOST tests/t.py::test_old -- superseded by test_new',
                                     'cites': ['bbbbbbb2']}]  # fmt: skip
    assert got['resolutions'][0]['cites'] == ['bbbbbbb2']


def test_git_intent_reads_commits_and_the_memory_the_range_adds(tmp_path):
    def git(*a):
        return subprocess.run(
            [shutil.which('git'), '-C', str(tmp_path), *a], check=True, capture_output=True, text=True
        ).stdout

    git('init', '-q')
    git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '--allow-empty', '-m', 'base')
    (tmp_path / '.claude/memory').mkdir(parents=True)
    (tmp_path / '.claude/memory/n.md').write_text('intent text', encoding='utf-8')
    (tmp_path / 'code.py').write_text('x = 1\n', encoding='utf-8')
    git('add', '.')
    git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'feat: n')
    intent = git_intent('HEAD~1..HEAD', str(tmp_path))
    assert [c['subject'] for c in intent['commits']] == ['feat: n']
    assert intent['memory'] == [{'path': '.claude/memory/n.md', 'text': 'intent text'}]


@pytest.fixture(scope='module')
def method(method_dir):
    return load_method(method_dir, principles=PRINCIPLES, registry=())


FINDING = {'severity': 'HIGH', 'security': False, 'rule': 'occams-razor', 'file': 'a.py', 'anchor': 'k = 1',
           'summary': 'S', 'category': 'Bug'}  # fmt: skip


def _merge(method, disposed):
    return merge(method, range_='a..b', profiles=['diff'],
                 reviewers=[{'name': 'diff', 'provider': 'anthropic', 'status': 'pending'}],
                 results=[{'name': 'diff', 'findings': [FINDING]}], disposed=disposed)  # fmt: skip


def test_the_id_ignores_whitespace_in_the_anchor(method):
    spec = method.spec['merge']['id']
    assert finding_id(spec, FINDING) == finding_id(spec, {**FINDING, 'anchor': '  k   =  1 '})


def test_a_disposed_id_is_not_re_raised_but_listed_with_its_reason(method):
    fid = finding_id(method.spec['merge']['id'], FINDING)
    doc = _merge(method, {fid: 'accepted: out of scope'})
    assert doc['findings'] == []
    assert doc['disposed'] == [{'id': fid, 'reason': 'accepted: out of scope'}]
    assert _errors(method.output_schema, doc) == []
    assert 'disposed' not in _merge(method, {})


def _ok_call(request):
    if 'findings' in request['schema']['properties']:
        return _reply({'findings': [FINDING]})
    return _reply({'changes': [{'claim': 'adds', 'cites': ['aaaaaaa1']}], 'open_items': []})


def test_the_cli_puts_the_summary_first_and_honours_disposed(method_dir, tmp_path, capsys):
    fid = finding_id(load_method(principles=PRINCIPLES, registry=()).spec['merge']['id'], FINDING)
    disposed = tmp_path / 'disposed.json'
    disposed.write_text(json.dumps({'disposed': [{'id': fid, 'reason': 'wontfix'}]}), encoding='utf-8')
    body = tmp_path / 'pr.md'
    body.write_text('PR says why', encoding='utf-8')
    seen, calls = [], []

    def intent(range_, repo):
        seen.append(range_)
        return {**INTENT, 'memory': []}

    argv = ['--principles', str(PRINCIPLES_FILE), '--rules', '', '--method', str(method_dir), '--range', 'a..b',
            '--model', 'm', '--disposed', str(disposed), '--pr-body', str(body)]  # fmt: skip
    assert main(argv, call=lambda r: calls.append(r) or _ok_call(r), numstat=lambda *a, **k: [(1, 0, 'k.py')],
                intent=intent) == 0  # fmt: skip
    doc = json.loads(capsys.readouterr().out)
    assert next(iter(doc)) == 'summary'
    assert doc['summary']['changes'] == [{'claim': 'adds', 'cites': ['aaaaaaa1']}]
    assert doc['findings'] == []
    assert doc['disposed'] == [{'id': fid, 'reason': 'wontfix'}]
    assert seen == ['a..b']
    assert any('PR says why' in c['context'][0]['text'] for c in calls)
    assert _errors(load_method(method_dir, principles=PRINCIPLES, registry=()).output_schema, doc) == []


def test_no_summary_skips_the_summary_call(method_dir, capsys):
    argv = ['--principles', str(PRINCIPLES_FILE), '--rules', '', '--method', str(method_dir), '--range', 'a..b',
            '--model', 'm', '--no-summary']  # fmt: skip
    assert main(argv, call=_ok_call, numstat=lambda *a, **k: [(1, 0, 'k.py')], intent=None) == 0
    assert 'summary' not in json.loads(capsys.readouterr().out)
