"""NEVER-LOG-SECRETS: no output sink in the tracked Python interpolates a credential's value."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.rules import tracked_files
from dev_lab.dev.secretlog import secret_interpolations

_ROOT: Final = Path(__file__).resolve().parents[2]


def test_no_sink_interpolates_a_secret() -> None:
    files = [rel for rel in tracked_files(_ROOT) if str(rel).endswith('.py')]
    assert files, 'the tracked Python population read EMPTY'
    found = [
        f'{rel}:{line} {name}'
        for rel in files
        for line, name in secret_interpolations((_ROOT / rel).read_text(encoding='utf-8'))
    ]
    assert not found, found


def test_the_scan_sees_a_planted_leak() -> None:
    assert secret_interpolations('print(f"{api_key}")\n')
