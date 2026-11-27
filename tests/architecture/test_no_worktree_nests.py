"""WORKTREES-DO-NOT-NEST: no linked worktree of this repo lies inside another linked worktree.

The reader and the remedy are the family's (dev_lab.dev.famtests.worktreeplace); this file supplies the root.
"""

from __future__ import annotations

from pathlib import Path

from dev_lab.dev.famtests.worktreeplace import assert_no_worktree_nests

ROOT = Path(__file__).resolve().parents[2]


def test_no_worktree_nests() -> None:
    assert_no_worktree_nests(root=ROOT)
