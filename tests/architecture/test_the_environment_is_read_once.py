"""ENV-IS-READ-ONCE: only the entry module ``[tool.dev_lab.env]`` declares reads the process environment."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev import envreads

_ROOT: Final = Path(__file__).resolve().parents[2]


def test_only_the_declared_entry_reads_the_environment() -> None:
    envreads.check(_ROOT)


def test_the_scan_sees_a_planted_read() -> None:
    assert envreads.python_reads('import os\nos.environ.get("X")\n')
