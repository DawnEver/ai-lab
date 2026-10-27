"""MERGE-OR-DELETE: this repo DECLARES its branch set (`[tool.dev_lab.branchset]`, `prune_merged = true`).

The LIVE branch set is the forge's `branchset` status, never a test (ruling 2026-10-24).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.branchset import declared_branchset
from dev_lab.dev.lanepolicy import load_policy

_ROOT: Final = Path(__file__).resolve().parents[2]


def test_this_repo_declares_the_one_pr_flow() -> None:
    """A change returns as a `pr/*` PR; integration moves only by the forge's merge of one.

    The two declarations ARE the rule: `ephemeral = ['pr/*']` admits a PR head with no declared session
    row, and `pr_merges_since` refuses a client push that adds a commit to integration.
    """
    assert load_policy(_ROOT).pr_merges_since, 'declare [tool.dev_lab.integrator] pr_merges_since'
    assert declared_branchset(_ROOT).is_ephemeral('pr/x'), "declare [tool.dev_lab.branchset] ephemeral = ['pr/*']"
