"""DECLARED-LAYOUT, NO-BACKUP-COPIES, ONE-FORMAT-PER-NEED: the whole tree is the allow-list in layout.toml."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.layout import declared_layout, judge, layout_violations

_ROOT: Final = Path(__file__).resolve().parents[2]


def test_every_directory_and_format_is_declared() -> None:
    _, found = layout_violations(_ROOT)
    assert not found, found


def test_the_allow_list_refuses_a_planted_format() -> None:
    assert judge('src/ai_lab/planted.yaml', declared_layout(_ROOT))
