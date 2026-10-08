"""NO-REFLECTION over this tree: capability is a declared record, never a probe."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import noreflection

_ROOT: Final = Path(__file__).resolve().parents[2]

#: The tracked Python files read; the floor refuses a walk that read almost nothing.
_FLOOR: Final = 30


def test_no_tracked_python_reflects() -> None:
    noreflection.assert_no_reflection(root=_ROOT, allowed={}, floor=_FLOOR)
