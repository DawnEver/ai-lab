"""Every retired spelling is REFUSED BY NAME where it would be read, and a scanned one appears nowhere else.

The registry is :data:`ai_lab.retired.RETIRED`. Each row is driven through the entry point its kind is
read at -- a ``connect`` spec, a user provider row's name, wire, field or auth -- and must be refused
naming its replacement. A scanned spelling found in any other tracked file is a leftover.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ai_lab import Unsupported, connect, providers
from ai_lab.retired import RETIRED, Retired, refuse

_ROOT = Path(__file__).resolve().parents[2]
#: The registry itself and this test are the only files a retired spelling may appear in; dated memory
#: (``.claude/``) is history, and history names what was true then.
_HOMES = frozenset({'src/ai_lab/retired.py', 'tests/architecture/test_retired_names_are_registered.py'})

_ROW = 'endpoint = "https://x.test/v1"\nwire = "openai_chat"\nauth = "local"\n'


def _user_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: str) -> None:
    monkeypatch.setenv('AI_LAB_HOME', str(tmp_path))
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'providers.toml').write_text(rows, encoding='utf-8')


def _row_using(retired: Retired) -> str:
    """A user provider row whose only defect is *retired*'s spelling, in the place its kind is read."""
    if retired.kind == 'provider':
        return f'[{retired.spelling}]\n{_ROW}'
    if retired.kind == 'wire':
        return f'[mine]\n{_ROW.replace("openai_chat", retired.spelling)}'
    if retired.kind == 'auth':
        return f'[mine]\n{_ROW.replace("local", retired.spelling)}'
    return f'[mine]\n{_ROW}{retired.spelling} = "x"\n'


@pytest.mark.parametrize('retired', RETIRED, ids=lambda r: f'{r.kind}:{r.spelling}')
def test_a_user_row_using_a_retired_spelling_is_refused_naming_the_replacement(retired, tmp_path, monkeypatch):
    _user_rows(tmp_path, monkeypatch, _row_using(retired))
    with pytest.raises(Unsupported, match=re.escape(f'use {retired.replacement!r}')):
        providers()


@pytest.mark.parametrize('retired', [r for r in RETIRED if r.kind == 'provider'], ids=lambda r: r.spelling)
def test_a_retired_provider_is_refused_at_connect(retired):
    with pytest.raises(Unsupported, match=re.escape(f'use {retired.replacement!r}')):
        connect(f'{retired.spelling}:m')


def test_the_planted_control_a_live_row_loads(tmp_path, monkeypatch):
    """THE PLANTED CONTROL: the same row with no retired spelling is accepted, so the refusals above are real."""
    _user_rows(tmp_path, monkeypatch, f'[mine]\n{_ROW}')
    assert providers()['mine'].wire == 'openai_chat'
    refuse('provider', 'mine', 'here')


def _leftovers(spelling: str, files: dict[str, str]) -> list[str]:
    word = re.compile(rf'(?<![A-Za-z0-9_]){re.escape(spelling)}(?![A-Za-z0-9_])')
    return sorted(
        path
        for path, text in files.items()
        if path not in _HOMES and not path.startswith('.claude/') and word.search(text)
    )


def _tracked() -> dict[str, str]:
    names = subprocess.run(
        [shutil.which('git') or 'git', 'ls-files'], cwd=_ROOT, capture_output=True, text=True, check=True, timeout=60
    ).stdout.split()
    files = {}
    for name in names:
        path = _ROOT / name
        if name.endswith(('.py', '.toml', '.md')) and path.is_file():
            files[name] = path.read_text(encoding='utf-8')
    return files


def test_a_scanned_retired_spelling_appears_in_no_other_tracked_file():
    files = _tracked()
    assert len(files) > 10, 'the scan read the tree'  # a floor: an empty scan proves nothing
    found = {r.spelling: hits for r in RETIRED if r.scanned and (hits := _leftovers(r.spelling, files))}
    assert not found, f'retired spellings outside ai_lab.retired: {found}'


def test_the_planted_control_the_scan_finds_a_leftover():
    assert _leftovers('claude_code', {'README.md': 'use `claude_code` here', 'x.py': 'claude_code_x'}) == ['README.md']
