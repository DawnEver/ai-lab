"""PUSH ADMISSION has one implementation, `dev_lab.dev.admission`; ai_lab keeps no local copy.

THE BODY IS THE FAMILY'S (`dev_lab.dev.famtests.localadmission`). This file holds only the
repo's answers: the trees walked and the read floor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.famtests import localadmission

_ROOT: Final = Path(__file__).resolve().parents[2]
_ROOTS: Final = (('src', '*.py'), ('tests', '*.py'))
#: Below the 2026-10-19 measurement on purpose: a floor refuses an UNREAD tree.
_FILES_READ_FLOOR: Final = 20


def test_the_scanner_still_convicts_a_planted_copy() -> None:
    localadmission.assert_the_scanner_still_convicts()


def test_no_local_admission_rule_is_kept() -> None:
    scan = localadmission.take_scan(_ROOT, roots=_ROOTS)
    localadmission.assert_no_local_admission(scan, floor=_FILES_READ_FLOOR)
