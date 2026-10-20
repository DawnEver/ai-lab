"""ai-lab ADOPTS the family rules registry: every rule is enforced here by a named mechanism, or absent.

``lab_commons.dev.rules`` holds the one canonical statement of each development rule; ``assert_adopted``
resolves every rule this repo claims against a TRACKED mechanism in this tree and refuses a rule that is
neither claimed nor declared absent. It proves the accounting, not that the mechanisms pass -- that is
the suite's business.

ai-lab is a LIBRARY: no push gate, no deny-hook engine, no vendor arbitration, no physical units, no
memory tree. A rule whose subject this repo does not have is declared absent rather than given a
mechanism that checks nothing, and the absent set may only shrink.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.profile import RepoProfile
from lab_commons.dev.rules import Adoption, LintRule, TestPath, assert_adopted

_ROOT: Final = Path(__file__).resolve().parents[2]

_PROFILE: Final = RepoProfile(app_name='ai_lab', package='ai_lab', root=_ROOT, lint_config=_ROOT / 'pyproject.toml')

_SURFACE: Final = TestPath('tests/architecture/test_the_public_surface_is_declared.py')

_MECHANISMS: Final = {
    # Capability is a declared Wire record; the shared scanner reads every tracked Python file.
    'NO-REFLECTION': (TestPath('tests/architecture/test_no_reflection.py'),),
    # Every family artefact is the live base plus an (empty) delta; nothing project-level is unaccounted.
    'PROJECT-FILES-HAVE-ONE-SOURCE': (TestPath('tests/architecture/test_the_family_config_is_rendered.py'),),
    # `__all__` everywhere, and one public name has one home.
    'PUBLIC-SURFACE-DECLARED': (_SURFACE,),
    'FIX-THE-CAUSE': (_SURFACE,),
    # The family lint set selects PLC, so a function-body import fails `ruff check`.
    'NO-LAZY-IMPORT': (LintRule(code='PLC0415'),),
    # An unknown provider or wire, a missing key, an image to a blind provider, a verb a wire lacks:
    # each RAISES Unsupported naming the remedy.
    'UNSUPPORTED-RAISES': (TestPath('tests/unit/test_client.py'),),
    # A missing key is refused naming the `auth login` command that stores it.
    'REFUSAL-NAMES-THE-REMEDY': (TestPath('tests/unit/test_client.py'),),
    # ai_lab.retired is the one registry; each row is refused by name at its entry point, and a scanned
    # spelling found in any other tracked file reds (a planted leftover must be found).
    'RETIRED-NAMES-REGISTERED': (TestPath('tests/architecture/test_retired_names_are_registered.py'),),
    # Adopted 2026-10-20 with the ledger's first schema change (schema 2 dropped usage.cost_usd).
    'FORMATS-UPGRADE-ON-READ': (TestPath('tests/unit/test_ledger_and_rules.py'),),
    # integrate/main and main are declared in pyproject.toml; the base table's `session-branch` pre-push
    # row refuses the three push-side violations, and the one writer is git's own (guarded from force by it).
    **dict.fromkeys(
        ('ONE-WRITER-PER-SESSION-BRANCH', 'PUBLISH-IS-THE-LOCAL-REF', 'SESSION-HISTORY-ONLY-GROWS',
         'INTEGRATION-IS-MERGES-ONLY'),
        (TestPath('pyproject.toml'), TestPath('tests/architecture/test_session_branch_pushes_are_guarded.py')),
    ),
    # Adopted 2026-10-19, as optimi-lab: `[tool.lab_commons.integrator]` declares integrate/main and main.
    # `forge sync` merges integrate/main only and `forge integration` retargets a PR opened against main
    # (both read that table); main moves only by `integrate promote`, whose admission is the kit's with no
    # local copy; MERGE-OR-DELETE is the branch-set apply over `prune_merged = true`, policed on origin.
    **dict.fromkeys(('DOWNSTREAM-SYNCS-INTEGRATION', 'PRS-TARGET-INTEGRATION'), (TestPath('pyproject.toml'),)),
    'TRUNK-MOVES-BY-FAST-FORWARD': (
        TestPath('pyproject.toml'), TestPath('tests/architecture/test_push_admission_is_delegated.py'),
    ),
    'MERGE-OR-DELETE': (
        TestPath('pyproject.toml'), TestPath('tests/architecture/test_the_branches_are_the_declared_set.py'),
    ),
    # Adopted 2026-10-19 with the lab-commons ef06ab3 relock: both are the forge's server-side steps
    # (lab_commons.dev.forgemerge, lab_commons.dev.forgereview) over every repo whose integrator table
    # declares an integration branch; ai-lab declares it, so its PRs merge and are reviewed by the kit alone.
    'INTEGRATION-TAKES-ONLY-FORGE-MERGES': (
        TestPath('pyproject.toml'), TestPath('tests/architecture/test_session_branch_pushes_are_guarded.py'),
    ),
    'PR-REVIEW-IS-ADVISORY': (TestPath('pyproject.toml'),),
}  # fmt: skip

#: Rules with no subject in this library today, BY NAME -- deriving it from ``RULES`` would silently
#: absorb a rule added upstream instead of making this repo decide about it. A rule leaves by gaining a
#: mechanism above; the set may only shrink.
_ABSENT: Final = frozenset({
    # Workflow machinery ai-lab does not run: no push gate, verdict ledger, forge door, lanes, sessions.
    'VERDICT-BAR-IS-THE-INCREMENT', 'NETWORK-RETRY-THEN-REPORT', 'SHARED-CHECKOUT', 'FORGE-THROUGH-THE-DOOR',
    'VERDICT-AS-STATUS', 'ISSUE-IS-INTENT', 'ONE-BRANCH-PER-SESSION', 'MERGE-DEVIATIONS-NAMED',
    'AUTO-MODE-RUNS-THE-DOORS', 'VERDICTS-FROM-THE-FORGE', 'MAIN-SESSION-PUBLISHES',
    'SUBAGENT-NO-PUSH', 'WORKTREES-STAY-INSIDE', 'ONE-BOX-ONE-LOCK',
    # The environment: no own venv yet (the agent guard refuses creating one until a door is declared),
    # so nothing installs hooks or dependencies through a checked door here.
    'HOOKS-ARE-WIRED', 'INSTALL-DOOR-DELIVERS-THE-DECLARATION', 'ENV-MUTATION-THROUGH-THE-DOOR',
    'AGENT-GUARD', 'LATEST-DEPENDENCIES',
    # Domain and measurement rules: no physics, units, tolerances, bars, capability matrix or cases.
    'IMPLEMENT-EVERYTHING', 'BAR-IS-A-CONSTANT', 'UNITS-GO-THROUGH-PINT', 'TOLERANCE-CARRIES-A-UNIT',
    'REGISTRY-OWNS-THE-DECISION', 'PRODUCTION-ENTRY-POINT',
    # Docs and memory: no rules pages, injected docs or memory tree in this repo.
    'DOCS-SPLIT', 'MEMORY-SHAPE', 'INJECTED-DOC-WIDTH-CEILING', 'INJECTED-TEXT-IS-PROGRESSIVE',
    'NO-CJK-IN-TRACKED-SOURCE', 'REPORTS-LIVE-IN-MEMORY',
    # Code placement and scratch: one src/ and one tests/ tree, no scratch.
    'CODE-IN-CODE-ROOTS', 'SCRATCH-ARCHIVED-OR-PROMOTED',
    # Meta-rules about guards: the guards above are the family's shared bodies, which carry them.
    'DECLARATION-LIES', 'PLANTED-CONTROL', 'FLOOR-ON-EVERY-SCAN', 'ESCAPE-HATCH-CEILING', 'RATCHET-TWO-SIDES',
    'NAMED-SETS-NOT-COUNTS', 'XFAIL-NOT-SKIP', 'MODULE-SIZE-ALARM',
})  # fmt: skip

#: The size of the set as of 2026-10-08 (41), re-based 2026-10-13 when the registry added six rules with
#: no subject here (the five integration-branch rules and REPORTS-LIVE-IN-MEMORY), and again (48) for
#: PUSH-FROM-ITS-OWN-BRANCH, an integration-line rule with no line here; lowering it is the only
#: legal edit. Lowered to 47 on 2026-10-19: the registry retired INTEGRATION-LIVES-IN-THE-MAIN-CHECKOUT and
#: PUSH-FROM-ITS-OWN-BRANCH and added FORMATS-UPGRADE-ON-READ. Lowered to 43 the same day: the four
#: branch-flow rules gained mechanisms once integrate/main was declared. Lowered to 42 on 2026-10-20:
#: RETIRED-NAMES-REGISTERED gained ai_lab.retired and its scan. Lowered to 41 the same day:
#: FORMATS-UPGRADE-ON-READ gained the ledger's declared upgrade steps.
_ABSENT_CEILING: Final = 41


def test_every_rule_is_enforced_or_declared_absent() -> None:
    assert_adopted(_PROFILE, Adoption(app_name='ai_lab', mechanisms=_MECHANISMS, declared_absent=_ABSENT))


def test_the_absent_set_does_not_grow() -> None:
    assert len(_ABSENT) <= _ABSENT_CEILING, sorted(_ABSENT)


def test_no_rule_is_both_enforced_and_absent() -> None:
    assert not (_ABSENT & frozenset(_MECHANISMS))
