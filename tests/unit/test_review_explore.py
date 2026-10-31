"""EXPLORATORY REVIEW (user ruling 2026-10-31): the reviewer gets the change's map and reads what it chooses.

A real git repository per test: the prompt never carries the diff, a reviewer's request is answered from git, and a
change too large to list file by file is mapped by directory and still reviewed.
"""

import json
import shutil
import subprocess
from pathlib import Path

from ai_lab.review.__main__ import main
from ai_lab.review.explore import MAX_LISTED_FILES, TURNS, change_map, numstat, serve
from tests.review_principles import PRINCIPLES_FILE

_GIT = shutil.which('git') or 'git'
SENTINEL = 'secret_sauce = compute_the_thing()'
FINDING = {'severity': 'LOW', 'security': False, 'rule': 'occams-razor', 'file': 'src/a.py',
           'anchor': SENTINEL, 'summary': 'S', 'category': 'Bug'}  # fmt: skip


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run([_GIT, '-c', 'user.name=t', '-c', 'user.email=t@t', '-C', str(repo), *args],
                          capture_output=True, text=True, encoding='utf-8', check=True)  # fmt: skip
    return done.stdout.strip()


def _repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    """A repository whose one commit past ``base`` adds *files*; the range."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    (repo / 'README').write_text('r\n', encoding='utf-8')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'base')
    base = _git(repo, 'rev-parse', 'HEAD')
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding='utf-8')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'the change')
    return repo, f'{base}..{_git(repo, "rev-parse", "HEAD")}'


def _reply(result):
    return {'schema_version': 1, 'ok': True, 'kind': '', 'result': result, 'usage': None, 'key_slot': 1, 'error': ''}


class Scripted:
    """A reviewer that answers each turn from *turns* in order (the last repeats), recording every request."""

    def __init__(self, *turns):
        self.turns, self.requests = list(turns), []

    def __call__(self, request):
        self.requests.append(request)
        return _reply(self.turns[min(len(self.requests), len(self.turns)) - 1])


def _review(repo: Path, range_: str, call, capsys) -> dict:
    argv = ['--principles', str(PRINCIPLES_FILE), '--rules', '', '--range', range_, '--model', 'm',
            '--repo', str(repo), '--lens', 'light', '--no-summary']  # fmt: skip
    assert main(argv, call=call) == 0
    return json.loads(capsys.readouterr().out)


def _prompt(request) -> str:
    return '\n'.join(part['text'] for part in request['context'])


def test_the_prompt_carries_the_map_and_never_the_diff(tmp_path, capsys):
    """PLANTED CONTROL: the sentinel is in the change -- asking for its diff returns it."""
    repo, range_ = _repo(tmp_path, {'src/a.py': f'{SENTINEL}\n', 'src/b.py': 'x = 1\ny = 2\n'})
    call = Scripted({'findings': []})
    doc = _review(repo, range_, call, capsys)
    (only,) = call.requests
    prompt = _prompt(only)
    assert SENTINEL not in prompt, 'the whole diff is never inlined'
    assert 'src/a.py' in prompt
    assert '+2 -0' in prompt
    assert 'src/' in prompt
    assert {r['status'] for r in doc['reviewers']} == {'ok'}
    assert SENTINEL in serve({'diff': ['src/a.py']}, range_=range_, repo=str(repo), rows=numstat(range_, str(repo)))


def test_the_reviewer_reads_what_it_asks_for_and_then_answers(tmp_path, capsys):
    repo, range_ = _repo(tmp_path, {'src/a.py': f'{SENTINEL}\n'})
    call = Scripted({'requests': {'diff': ['src/a.py'], 'grep': ['compute_the_thing']}}, {'findings': [FINDING]})
    doc = _review(repo, range_, call, capsys)
    first, second = call.requests
    assert SENTINEL not in _prompt(first)
    assert SENTINEL in _prompt(second), 'the next turn carries what it asked for'
    assert 'src/a.py:1:' in _prompt(second), 'grep runs over the head'
    assert 'findings' not in first['schema'].get('required', []), 'an asking turn need not answer'
    assert [f['anchor'] for f in doc['findings']] == [SENTINEL]


def test_the_last_turn_must_answer(tmp_path, capsys):
    repo, range_ = _repo(tmp_path, {'src/a.py': f'{SENTINEL}\n'})
    call = Scripted({'requests': {'show': ['src/a.py']}})
    doc = _review(repo, range_, call, capsys)
    assert len(call.requests) == TURNS
    assert call.requests[-1]['schema']['required'] == ['findings']
    assert 'requests' not in call.requests[-1]['schema']['properties']
    assert {r['status'] for r in doc['reviewers']} == {'unavailable'}, 'a reviewer that never answered is no review'


def test_a_large_change_is_mapped_by_directory_and_still_reviewed(tmp_path, capsys):
    files = {f'pkg{d}/mod{n}.py': f'v = {n}\n' for d in range(8) for n in range(MAX_LISTED_FILES // 4)}
    files['pkg0/mod0.py'] = f'{SENTINEL}\n'
    repo, range_ = _repo(tmp_path, files)
    call = Scripted({'requests': {'list': ['pkg0']}}, {'requests': {'diff': ['pkg0/mod0.py']}}, {'findings': []})
    doc = _review(repo, range_, call, capsys)
    first = _prompt(call.requests[0])
    assert 'pkg3/' in first
    assert 'pkg3/mod7.py' not in first, 'past the bound the map is the directory level'
    assert f'past {MAX_LISTED_FILES}' in first
    assert 'pkg0/mod5.py' in _prompt(call.requests[1]), 'list opens a directory'
    assert SENTINEL in _prompt(call.requests[2])
    assert {r['status'] for r in doc['reviewers']} == {'ok'}
    assert first.count('mod') == 0, 'no file of the change is listed: the reviewer opens the directories it chooses'


def test_the_map_follows_a_rename_to_its_head_path(tmp_path):
    repo, range_ = _repo(tmp_path, {'old/name.py': 'a = 1\n' * 20})
    (repo / 'new').mkdir()
    _git(repo, 'mv', 'old/name.py', 'new/name.py')
    _git(repo, 'commit', '-q', '-m', 'move')
    moved = f'{range_.split("..")[1]}..HEAD'
    assert [row[2] for row in numstat(moved, str(repo))] == ['new/name.py']
    assert 'new/name.py' in change_map(moved, numstat(moved, str(repo)))
