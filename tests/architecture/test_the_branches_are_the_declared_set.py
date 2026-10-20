"""MERGE-OR-DELETE: origin and this box carry only `[tool.lab_commons.branchset]` (`prune_merged = true`).

THE BODY IS THE FAMILY'S (`lab_commons.dev.famtests.branchset`); this file supplies the trunk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from lab_commons.dev.branchset import declared_branchset
from lab_commons.dev.famtests import branchset
from lab_commons.dev.lanepolicy import load_policy

_ROOT: Final = Path(__file__).resolve().parents[2]
TRUNK: Final = 'main'


def test_origin_carries_only_the_declared_branches() -> None:
    branchset.assert_origin_branches_declared(root=_ROOT, branchset=declared_branchset(_ROOT))


@pytest.mark.workstation
def test_this_box_carries_only_the_declared_branches() -> None:
    """An undeclared local branch origin already holds reds; one holding unpushed work is reported."""
    branchset.assert_local_branches_declared(root=_ROOT, branchset=declared_branchset(_ROOT))


def test_the_branchset_guard_fires(tmp_path: Path) -> None:
    """PLANTED CONTROL, the family's: a stray branch on origin and locally, and unpushed work."""
    branchset.assert_the_planted_branchset_is_policed(tmp_path, trunk=TRUNK)


def test_this_repo_declares_the_one_pr_flow() -> None:
    """A change returns as a `pr/*` PR; integration moves only by the forge's merge of one.

    The two declarations ARE the rule: `ephemeral = ['pr/*']` admits a PR head with no declared session
    row, and `pr_merges_since` refuses a client push that adds a commit to integration.
    """
    assert load_policy(_ROOT).pr_merges_since, 'declare [tool.lab_commons.integrator] pr_merges_since'
    assert declared_branchset(_ROOT).is_ephemeral('pr/x'), "declare [tool.lab_commons.branchset] ephemeral = ['pr/*']"
