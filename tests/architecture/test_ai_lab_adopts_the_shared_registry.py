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
    # Every wait has a ceiling and a CLI past it is ended as a TREE from its root pid (a planted
    # grandchild must be dead afterwards).
    'REFUSAL-NAMES-THE-REMEDY': (TestPath('tests/unit/test_transport.py'),),
}

#: Rules with no subject in this library today, BY NAME -- deriving it from ``RULES`` would silently
#: absorb a rule added upstream instead of making this repo decide about it. A rule leaves by gaining a
#: mechanism above; the set may only shrink.
_ABSENT: Final = frozenset({
    # Workflow machinery ai-lab does not run: no push gate, verdict ledger, forge door, lanes, sessions.
    'VERDICT-BAR-IS-THE-INCREMENT', 'NETWORK-RETRY-THEN-REPORT', 'SHARED-CHECKOUT', 'FORGE-THROUGH-THE-DOOR',
    'VERDICT-AS-STATUS', 'ISSUE-IS-INTENT', 'ONE-BRANCH-PER-SESSION', 'MERGE-DEVIATIONS-NAMED',
    'AUTO-MODE-RUNS-THE-DOORS', 'ONE-RUN-AFTER-INTEGRATION', 'MAIN-SESSION-PUBLISHES',
    'SUBAGENT-NO-HEAVY-NO-PUSH', 'WORKTREES-STAY-INSIDE', 'ONE-BOX-ONE-LOCK',
    # The environment: no own venv yet (the agent guard refuses creating one until a door is declared),
    # so nothing installs hooks or dependencies through a checked door here.
    'HOOKS-ARE-WIRED', 'INSTALL-DOOR-DELIVERS-THE-DECLARATION', 'ENV-MUTATION-THROUGH-THE-DOOR',
    'AGENT-GUARD', 'LATEST-DEPENDENCIES',
    # Domain and measurement rules: no physics, units, tolerances, bars, capability matrix or cases.
    'IMPLEMENT-EVERYTHING', 'BAR-IS-A-CONSTANT', 'UNITS-GO-THROUGH-PINT', 'TOLERANCE-CARRIES-A-UNIT',
    'REGISTRY-OWNS-THE-DECISION', 'PRODUCTION-ENTRY-POINT',
    # Docs and memory: no rules pages, injected docs or memory tree in this repo.
    'DOCS-SPLIT', 'MEMORY-SHAPE', 'INJECTED-DOC-WIDTH-CEILING', 'INJECTED-TEXT-IS-PROGRESSIVE',
    'RETIRED-NAMES-REGISTERED', 'NO-CJK-IN-TRACKED-SOURCE',
    # Code placement and scratch: one src/ and one tests/ tree, no scratch.
    'CODE-IN-CODE-ROOTS', 'SCRATCH-ARCHIVED-OR-PROMOTED',
    # Meta-rules about guards: the guards above are the family's shared bodies, which carry them.
    'DECLARATION-LIES', 'PLANTED-CONTROL', 'FLOOR-ON-EVERY-SCAN', 'ESCAPE-HATCH-CEILING', 'RATCHET-TWO-SIDES',
    'NAMED-SETS-NOT-COUNTS', 'XFAIL-NOT-SKIP', 'MODULE-SIZE-ALARM',
})  # fmt: skip

#: The size of the set as of 2026-10-08; lowering it is the only legal edit.
_ABSENT_CEILING: Final = 41


def test_every_rule_is_enforced_or_declared_absent() -> None:
    assert_adopted(_PROFILE, Adoption(app_name='ai_lab', mechanisms=_MECHANISMS, declared_absent=_ABSENT))


def test_the_absent_set_does_not_grow() -> None:
    assert len(_ABSENT) <= _ABSENT_CEILING, sorted(_ABSENT)


def test_no_rule_is_both_enforced_and_absent() -> None:
    assert not (_ABSENT & frozenset(_MECHANISMS))
