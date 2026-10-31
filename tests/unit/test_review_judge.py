"""RE-JUDGE (user ruling 2026-10-31): every open finding is judged again against the new head, each claim adjudicated.

A real git repository per test and a scripted ``call``: one verdict per finding in order, a judge that reads a file
sees it, a failed call fails closed (``present``), a claim reaches the prompt and the diff never does.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from ai_lab.review.judge import main
from tests.review_principles import PRINCIPLES_FILE
from tests.unit.test_review_explore import _repo

SENTINEL = 'secret_sauce = compute_the_thing()'
CLAIM = 'the helper is called from exactly one place, so inlining it is the simpler side'


def _finding(fid: str, claim: str = '') -> dict:
    return {'id': fid, 'severity': 'LOW', 'security': False, 'rule': 'occams-razor', 'file': 'src/a.py',
            'anchor': 'compute', 'summary': f'summary {fid}', 'detail': 'D', 'suggestion': 'S',
            'claim': claim}  # fmt: skip


def _run(tmp_path: Path, findings: list[dict], call, capsys) -> tuple[int, dict | None, str]:
    repo, range_ = _repo(tmp_path, {'src/a.py': SENTINEL + '\n'})
    file = tmp_path / 'findings.json'
    file.write_text(json.dumps({'findings': findings}), encoding='utf-8')
    code = main(['--range', range_, '--repo', str(repo), '--model', 'm', '--rules', 'R1,R2',
                 '--principles', str(PRINCIPLES_FILE), '--findings', str(file)], call=call)  # fmt: skip
    out = capsys.readouterr()
    return code, (json.loads(out.out) if out.out else None), out.err


def _ok(result: Mapping) -> dict:
    return {'ok': True, 'result': dict(result), 'usage': {}}


def test_one_verdict_per_finding_in_input_order(tmp_path: Path, capsys) -> None:
    outcomes = iter(['fixed', 'false-positive', 'present'])
    requests: list[Mapping] = []

    def call(request: Mapping) -> Mapping:
        requests.append(request)
        return _ok({'outcome': next(outcomes), 'evidence': 'src/a.py:1'})

    code, doc, _ = _run(tmp_path, [_finding('a'), _finding('b'), _finding('c')], call, capsys)
    assert code == 0
    assert doc == {'schema_version': 1, 'verdicts': [
        {'id': 'a', 'outcome': 'fixed', 'evidence': 'src/a.py:1'},
        {'id': 'b', 'outcome': 'false-positive', 'evidence': 'src/a.py:1'},
        {'id': 'c', 'outcome': 'present', 'evidence': 'src/a.py:1'},
    ]}  # fmt: skip
    assert requests[0]['task'] == 'pr-review'
    assert requests[0]['provider'] == 'anthropic'
    assert 'requests' in requests[0]['schema']['properties']
    assert 'outcome' in requests[0]['schema']['properties']


def test_a_judge_that_asks_to_show_the_file_reads_it_next_turn(tmp_path: Path, capsys) -> None:
    prompts: list[str] = []

    def call(request: Mapping) -> Mapping:
        prompts.append('\n'.join(block['text'] for block in request['context']))
        if len(prompts) == 1:
            return _ok({'requests': {'show': ['src/a.py']}})
        assert set(request['schema']['properties']) >= {'outcome', 'evidence', 'requests'}
        return _ok({'outcome': 'fixed', 'evidence': 'src/a.py:1 no longer has it'})

    code, doc, _ = _run(tmp_path, [_finding('a')], call, capsys)
    assert code == 0
    assert doc['verdicts'] == [{'id': 'a', 'outcome': 'fixed', 'evidence': 'src/a.py:1 no longer has it'}]
    assert SENTINEL not in prompts[0]
    assert SENTINEL in prompts[1]


@pytest.mark.parametrize(
    ('reply', 'reason'),
    [
        ({'ok': False, 'kind': 'network', 'error': 'down'}, 'network'),
        (_ok({'outcome': 'maybe', 'evidence': 'x'}), 'no valid outcome'),
    ],
)
def test_a_failed_or_invalid_judgement_fails_closed(tmp_path: Path, capsys, reply: Mapping, reason: str) -> None:
    code, doc, _ = _run(tmp_path, [_finding('a')], lambda _request: reply, capsys)
    assert code == 0
    (verdict,) = doc['verdicts']
    assert verdict['outcome'] == 'present'
    assert verdict['evidence'].startswith('not judged: ')
    assert reason in verdict['evidence']


def test_the_claim_reaches_the_prompt_and_the_diff_never_does(tmp_path: Path, capsys) -> None:
    prompts: list[str] = []

    def call(request: Mapping) -> Mapping:
        prompts.append('\n'.join(block['text'] for block in request['context']))
        return _ok({'outcome': 'present', 'evidence': 'e'})

    _run(tmp_path, [_finding('a', CLAIM)], call, capsys)
    (prompt,) = prompts
    assert CLAIM in prompt
    assert 'summary a' in prompt
    assert 'occams-razor' in prompt
    assert 'src/a.py' in prompt
    assert SENTINEL not in prompt
    assert '@@' not in prompt


@pytest.mark.parametrize('content', ['not json', '{"findings": [{"id": "a"}]}', '{"other": []}'])
def test_a_bad_findings_file_is_refused(tmp_path: Path, capsys, content: str) -> None:
    repo, range_ = _repo(tmp_path, {'src/a.py': SENTINEL + '\n'})
    file = tmp_path / 'findings.json'
    file.write_text(content, encoding='utf-8')
    code = main(['--range', range_, '--repo', str(repo), '--model', 'm', '--rules', 'R1',
                 '--principles', str(PRINCIPLES_FILE), '--findings', str(file)],
                call=lambda _r: pytest.fail('no call on a refused invocation'))  # fmt: skip
    out = capsys.readouterr()
    assert code == 2
    assert out.out == ''
    assert 'refused' in out.err


def test_a_range_git_cannot_read_is_refused(tmp_path: Path, capsys) -> None:
    repo, _ = _repo(tmp_path, {'src/a.py': SENTINEL + '\n'})
    file = tmp_path / 'findings.json'
    file.write_text(json.dumps({'findings': [_finding('a')]}), encoding='utf-8')
    code = main(['--range', 'nope..nada', '--repo', str(repo), '--model', 'm', '--rules', 'R1',
                 '--principles', str(PRINCIPLES_FILE), '--findings', str(file)], call=lambda _r: {})  # fmt: skip
    assert code == 2
    assert capsys.readouterr().out == ''
