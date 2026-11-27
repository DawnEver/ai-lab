"""EVIDENCE-OUTLIVES-OUTPUT: no tracked value names an ignored path as its evidence.

The scan, its escapes ([tool.dev_lab.evidence_refs] in pyproject.toml) and the planted control are the
family's (dev_lab.dev.famtests.evidencerefs); this file supplies the repo root and the measured floor.
"""

from __future__ import annotations

from pathlib import Path

from dev_lab.dev.famtests.evidencerefs import (
    assert_no_tracked_value_cites_output_evidence,
    assert_the_evidence_scan_still_convicts,
)

ROOT = Path(__file__).resolve().parents[2]
#: Tracked code and data files the scan read on 2026-11-27 (148), less headroom for deletions.
FLOOR = 130
HEADROOM = 40


def test_no_tracked_value_cites_output_evidence() -> None:
    assert_no_tracked_value_cites_output_evidence(root=ROOT, floor=FLOOR, headroom=HEADROOM)


def test_planted_the_evidence_scan_still_convicts(tmp_path: Path) -> None:
    assert_the_evidence_scan_still_convicts(tmp_path)
