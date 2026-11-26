"""NEVER-LOG-SECRETS: no output sink in the tracked Python interpolates a credential's value."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

from dev_lab.dev.secretlog import secret_interpolations

_ROOT: Final = Path(__file__).resolve().parents[2]


def _tracked_python() -> list[Path]:
    out = subprocess.run(['git', 'ls-files', '*.py'], cwd=_ROOT, capture_output=True, text=True, check=True).stdout
    return [_ROOT / rel for rel in out.split()]


def test_no_sink_interpolates_a_secret() -> None:
    files = _tracked_python()
    assert files, 'the tracked Python population read EMPTY'
    found = [f'{p.relative_to(_ROOT)}:{line} {name}' for p in files
             for line, name in secret_interpolations(p.read_text(encoding='utf-8'))]
    assert not found, found


def test_the_scan_sees_a_planted_leak() -> None:
    assert secret_interpolations('print(f"{api_key}")\n')
