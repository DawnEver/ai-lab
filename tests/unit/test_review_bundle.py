"""``--bundle <dir>``: a review and a judge read a review bundle's plain files, and never start git.

The bundle is built here the way dev-lab's ``reviewbundle`` lays it out (``base/``, ``head/``, ``diff.patch``,
``bundle.json``) from a real repository, so each bundle-mode run is compared against the ``--range`` run of the
same change: same inputs, same prompts, same findings and verdicts.
"""

import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ai_lab.review.__main__ import main as review_main
from ai_lab.review.explore import numstat
from ai_lab.review.judge import main as judge_main
from ai_lab.review.source import BundleSource, GitSource
from tests.review_principles import PRINCIPLES_FILE
from tests.unit.test_review_explore import FINDING, SENTINEL, Scripted, _git

_EXE = shutil.which('git') or 'git'
PNG = b'\x89PNG\r\n\x1a\n\x00\x01binary'


def _change(tmp_path: Path) -> tuple[Path, str, str]:
    """A repository whose head adds, modifies, deletes and adds a binary file; ``(repo, base, head)``."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    for path, text in {'src/a.py': 'compute = 0\n', 'src/gone.py': 'old = 1\n', 'README': 'r\n'}.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding='utf-8')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'base')
    base = _git(repo, 'rev-parse', 'HEAD')
    (repo / 'src/a.py').write_text(f'compute = 1\n{SENTINEL}\n', encoding='utf-8')
    (repo / 'src/new.py').write_text('fresh = 2\n', encoding='utf-8')
    (repo / '.claude/memory').mkdir(parents=True)
    (repo / '.claude/memory/why.md').write_text('the intent', encoding='utf-8')
    (repo / 'img.png').write_bytes(PNG)
    (repo / 'src/gone.py').unlink()
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', 'the change')
    return repo, base, _git(repo, 'rev-parse', 'HEAD')


def _blob(repo: Path, rev: str, path: str) -> bytes | None:
    done = subprocess.run([_EXE, '-C', str(repo), 'show', f'{rev}:{path}'], capture_output=True, check=False)
    return done.stdout if done.returncode == 0 else None


def _bundle(repo: Path, base: str, head: str, out: Path) -> Path:
    """The review bundle of ``base..head``, laid out as dev-lab's ``reviewbundle`` materializes it."""
    names = _git(repo, 'diff', '--name-only', '--no-renames', base, head).splitlines()
    patch = subprocess.run([_EXE, '-C', str(repo), 'diff', '--no-renames', '--full-index', base, head],
                           capture_output=True, check=True).stdout.decode('utf-8')  # fmt: skip
    files = []
    for path in sorted(names):
        sides = {side: _blob(repo, rev, path) for side, rev in (('base', base), ('head', head))}
        binary = any(data is not None and b'\0' in data for data in sides.values())
        row = {'path': path, 'base_blob': None, 'head_blob': None, 'binary': binary, 'base': None, 'head': None}
        for side, data in sides.items():
            if data is None:
                continue
            row[f'{side}_blob'] = 'x' * 40
            if not binary:
                row[side] = base64.b64encode(data).decode('ascii')
                (out / side / path).parent.mkdir(parents=True, exist_ok=True)
                (out / side / path).write_bytes(data)
        files.append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'diff.patch').write_text(patch, encoding='utf-8', newline='')
    doc = {'version': 1, 'repo': 'owner/repo', 'pr': 7, 'base_ref': 'main', 'base_sha': base, 'head_sha': head,
           'diff': patch, 'declaration': '', 'files': files}  # fmt: skip
    (out / 'bundle.json').write_text(json.dumps(doc), encoding='utf-8')
    return out


@pytest.fixture
def change(tmp_path):
    repo, base, head = _change(tmp_path)
    return repo, f'{base}..{head}', _bundle(repo, base, head, tmp_path / 'bundle')


@pytest.fixture
def no_git(monkeypatch):
    """Every way to start git raises: a bundle-mode run that starts one fails the test."""

    def refuse(*_a, **_k):
        pytest.fail('a bundle-mode run started a process')

    monkeypatch.setattr(subprocess, 'run', refuse)
    monkeypatch.setattr(subprocess, 'Popen', refuse)
    monkeypatch.setattr(shutil, 'which', refuse)


def test_a_bundle_reads_what_git_reads_of_the_change(change):
    repo, range_, bundle = change
    git, read = GitSource(range_, str(repo)), BundleSource(bundle)
    assert read.label == range_
    assert read.numstat() == numstat(range_, str(repo)) == git.numstat()
    assert read.numstat(('src',)) == git.numstat(('src',))
    assert read.diff(('src/a.py',)) == git.diff(('src/a.py',))
    assert read.answer('show', 'src/a.py') == git.answer('show', 'src/a.py')
    assert SENTINEL in read.answer('grep', 'compute_the_thing')
    assert read.head_text('src/gone.py') is None, 'a deleted file is stated absent'
    assert read.intent()['memory'] == git.intent()['memory'] == [{'path': '.claude/memory/why.md',
                                                                    'text': 'the intent'}]  # fmt: skip
    assert read.intent()['commits'] == [], 'a bundle carries no commits'


def test_a_file_the_bundle_does_not_hold_is_said_so_never_called_absent(change):
    _repo, _range, bundle = change
    read = BundleSource(bundle)
    assert 'not in the review bundle' in read.answer('show', 'README')
    with pytest.raises(LookupError):
        read.head_text('README')
    with pytest.raises(LookupError):
        read.head_text('img.png')


def _review(argv: list[str], call, capsys) -> dict:
    base = ['--principles', str(PRINCIPLES_FILE), '--rules', '', '--model', 'm', '--lens', 'light', '--no-summary']
    assert review_main([*base, *argv], call=call) == 0
    return json.loads(capsys.readouterr().out)


def _prompts(call: Scripted) -> list[str]:
    return ['\n'.join(part['text'] for part in request['context']) for request in call.requests]


def test_a_bundle_review_sees_what_the_range_review_sees_and_starts_no_git(change, capsys, monkeypatch, request):
    repo, range_, bundle = change
    turns = ({'requests': {'diff': ['src/a.py'], 'show': ['src/new.py'], 'grep': ['compute_the_thing']}},
             {'findings': [FINDING]})  # fmt: skip
    by_range = Scripted(*turns)
    ranged = _review(['--range', range_, '--repo', str(repo)], by_range, capsys)
    request.getfixturevalue('no_git')
    by_bundle = Scripted(*turns)
    bundled = _review(['--bundle', str(bundle)], by_bundle, capsys)
    assert _prompts(by_bundle) == _prompts(by_range)
    assert bundled['findings'] == ranged['findings'] and bundled['range'] == ranged['range']


def test_a_bundle_summary_reads_the_pr_body_and_the_memory_and_starts_no_git(change, capsys, no_git, tmp_path):
    _repo, _range, bundle = change
    body = tmp_path / 'body.md'
    body.write_text('PR BODY TEXT', encoding='utf-8')
    seen = []

    def call(req):
        seen.append(req)
        if req.get('tag') == 'summary':
            return {'ok': True, 'result': {'changes': [], 'open_items': []}, 'usage': {}}
        return {'ok': True, 'result': {'findings': []}, 'usage': {}}

    argv = ['--principles', str(PRINCIPLES_FILE), '--rules', '', '--model', 'm', '--lens', 'light',
            '--bundle', str(bundle), '--pr-body', str(body)]  # fmt: skip
    assert review_main(argv, call=call) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['summary']['sources'] == {'commits': [], 'memory': ['.claude/memory/why.md']}
    (summary,) = [r for r in seen if r.get('tag') == 'summary']
    assert 'PR BODY TEXT' in summary['context'][0]['text']


@pytest.mark.parametrize('main', [review_main, judge_main])
def test_range_and_bundle_are_exactly_one(main, change, tmp_path):
    _repo, range_, bundle = change
    common = ['--model', 'm', '--rules', '', '--principles', str(PRINCIPLES_FILE), '--findings', str(tmp_path / 'f')]
    with pytest.raises(SystemExit) as both:
        main([*common, '--range', range_, '--bundle', str(bundle)])
    with pytest.raises(SystemExit) as neither:
        main(common)
    assert both.value.code == neither.value.code == 2


def _judge(argv: list[str], findings: Path, call, capsys) -> dict:
    code = judge_main([*argv, '--model', 'm', '--rules', 'R1', '--principles', str(PRINCIPLES_FILE),
                       '--findings', str(findings)], call=call)  # fmt: skip
    assert code == 0
    return json.loads(capsys.readouterr().out)


def test_a_bundle_judge_reaches_the_range_judges_verdicts_and_starts_no_git(change, capsys, tmp_path, request):
    repo, range_, bundle = change
    findings = tmp_path / 'findings.json'
    rows = [{'id': fid, 'severity': 'HIGH', 'security': False, 'rule': 'occams-razor', 'file': file,
             'anchor': 'compute', 'summary': 's', 'detail': 'd', 'suggestion': 's', 'claim': '',
             'reviewers': ['diff@anthropic']} for fid, file in (('F1', 'src/a.py'), ('F2', 'src/gone.py'),
                                                                 ('F3', 'README'))]  # fmt: skip
    findings.write_text(json.dumps({'findings': rows}), encoding='utf-8')
    prompts: dict[str, list[str]] = {'range': [], 'bundle': []}

    def scripted(mode):
        def call(req):
            prompts[mode].append('\n'.join(part['text'] for part in req['context']))
            return {'ok': True, 'result': {'outcome': 'fixed', 'evidence': 'e', 'cites': ['src/a.py:1']},
                    'usage': {}}  # fmt: skip

        return call

    ranged = _judge(['--range', range_, '--repo', str(repo)], findings, scripted('range'), capsys)
    request.getfixturevalue('no_git')
    bundled = _judge(['--bundle', str(bundle)], findings, scripted('bundle'), capsys)
    assert [v['outcome'] for v in ranged['verdicts']] == ['fixed'] * 3
    assert bundled['verdicts'] == ranged['verdicts']
    masked = {mode: [re.sub('[0-9a-f]{24}', 'N', text) for text in texts] for mode, texts in prompts.items()}
    assert masked['bundle'][:4] == masked['range'][:4], 'F1 and F2: the same prompts, sample by sample'
    assert 'does NOT exist at the head' in masked['bundle'][2]
    assert 'not in the review bundle' in masked['bundle'][4], 'an unchanged file is never called absent'
