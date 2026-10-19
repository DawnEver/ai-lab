"""Every push of a session branch runs the kit's session-branch invariants at pre-push.

PUBLISH-IS-THE-LOCAL-REF, SESSION-HISTORY-ONLY-GROWS and INTEGRATION-IS-MERGES-ONLY are refused by the
base table's ``session-branch`` row (``lab_commons/dev/githooks/base.toml``) over the branches
``[tool.lab_commons.branchset]`` and ``[tool.lab_commons.integrator]`` declare; ONE-WRITER-PER-SESSION-BRANCH
is git's own worktree exclusivity plus push compare-and-swap, guarded from force by the same row. This pins
that ai-lab's merged hook table keeps the row and runs it on every ref.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.githooks.runner import hook_table

_ROOT: Final = Path(__file__).resolve().parents[2]


def test_every_push_runs_the_session_branch_invariants() -> None:
    rows = {hook.id: hook for hook in hook_table(_ROOT)['pre-push']}
    assert 'session-branch' in rows, sorted(rows)
    assert list(rows['session-branch'].run)[-1:] == ['lab_commons.dev.sessionbranch']
    assert rows['session-branch'].always_run
