"""Every conflicted merge declares its resolutions at commit (MERGE-RESOLUTION-DECLARED).

The base table's commit-msg ``merge-resolution`` row (``dev_lab/dev/githooks/base.toml``) recomputes the
conflicted set and refuses a merge commit that does not declare each file as ``<path>: <class> -- <why>``.
This pins that ai-lab's merged hook table keeps the row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.githooks.runner import hook_table

_ROOT: Final = Path(__file__).resolve().parents[2]


def test_every_merge_commit_runs_the_resolution_check() -> None:
    rows = {hook.id: hook for hook in hook_table(_ROOT)['commit-msg']}
    assert 'merge-resolution' in rows, sorted(rows)
    assert list(rows['merge-resolution'].run)[-1:] == ['dev_lab.dev.mergeresolution']
