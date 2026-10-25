"""``ai-lab <command>``: the machine install's one entry point, so a host names no interpreter path."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from ai_lab import cli

_PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'


def test_the_install_declares_the_ai_lab_command() -> None:
    scripts = tomllib.loads(_PYPROJECT.read_text(encoding='utf-8'))['project']['scripts']
    assert scripts['ai-lab'] == 'ai_lab.cli:main'


def test_review_passes_its_arguments_through(monkeypatch) -> None:
    seen = []
    monkeypatch.setitem(cli.COMMANDS, 'review', lambda argv: seen.append(argv) or 0)
    assert cli.main(['review', '--range', 'a..b']) == 0
    assert seen == [['--range', 'a..b']]


def test_the_command_exit_code_is_returned(monkeypatch) -> None:
    monkeypatch.setitem(cli.COMMANDS, 'review', lambda _argv: 2)
    assert cli.main(['review']) == 2


@pytest.mark.parametrize('argv', [[], ['nonsense']])
def test_an_unknown_or_missing_command_exits_2_naming_the_known_ones(argv, capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(argv)
    assert stop.value.code == 2
    assert 'review' in capsys.readouterr().err
