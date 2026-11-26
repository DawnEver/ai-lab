"""ai-lab ADOPTS the family rules registry: every rule is enforced here by a named mechanism, or absent.

``dev_lab.dev.rules`` holds the one canonical statement of each development rule; ``assert_adopted``
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

from dev_lab.dev.profile import RepoProfile
from dev_lab.dev.rules import Adoption, LintRule, TestPath, assert_adopted

_ROOT: Final = Path(__file__).resolve().parents[2]

_PROFILE: Final = RepoProfile(app_name='ai_lab', package='ai_lab', root=_ROOT, lint_config=_ROOT / 'pyproject.toml')

_SURFACE: Final = TestPath('tests/architecture/test_the_public_surface_is_declared.py')

_MECHANISMS: Final = {
    # The base commit-msg `merge-resolution` row refuses a conflicted merge that does not declare each file.
    'MERGE-RESOLUTION-DECLARED': (TestPath('tests/architecture/test_merges_declare_their_resolutions.py'),),
    # Capability is a declared Wire record; the shared scanner reads every tracked Python file.
    'NO-REFLECTION': (TestPath('tests/architecture/test_no_reflection.py'),),
    # Every family artifact is the live base plus an (empty) delta; nothing project-level is unaccounted.
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
    'FORMATS-MIGRATE-ONCE': (TestPath('tests/unit/test_ledger_and_rules.py'),),
    # integrate/main and main are declared in pyproject.toml; the base table's `session-branch` pre-push
    # row refuses the three push-side violations, and the one writer is git's own (guarded from force by it).
    **dict.fromkeys(
        ('ONE-WRITER-PER-SESSION-BRANCH', 'PUBLISH-IS-THE-LOCAL-REF', 'SESSION-HISTORY-ONLY-GROWS',
         'INTEGRATION-IS-MERGES-ONLY', 'ORIGIN-HOLDS-ONLY-DECLARED-BRANCHES'),
        (TestPath('pyproject.toml'), TestPath('tests/architecture/test_session_branch_pushes_are_guarded.py')),
    ),
    # Adopted 2026-10-19, as optimi-lab: `[tool.dev_lab.integrator]` declares integrate/main and main.
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
    # (dev_lab.dev.forgemerge, dev_lab.dev.forgereview) over every repo whose integrator table
    # declares an integration branch; ai-lab declares it, so its PRs merge and are reviewed by the kit alone.
    'INTEGRATION-TAKES-ONLY-FORGE-MERGES': (
        TestPath('pyproject.toml'), TestPath('tests/architecture/test_session_branch_pushes_are_guarded.py'),
    ),
    'PR-REVIEW-IS-ONE-SHOT-ADVICE': (TestPath('pyproject.toml'),),
    # Adopted 2026-10-23 with the move to dev-lab: the PR-flow rows are the forge's own admission and carry
    # steps (dev_lab.dev.pradmit, dev_lab.dev.forgecarry) over the same declared integrator table.
    'ONE-SESSION-ONE-PR': (TestPath('pyproject.toml'),),
    'NO-STACKED-PRS': (TestPath('pyproject.toml'),),
    'DOCS-ONLY-CARRIES': (TestPath('pyproject.toml'),),
    # Adopted 2026-11-03 with dev-lab integrate/main: three more PR-flow rows of the forge's own admission over
    # the same declared integrator table (a PR opens as a draft, forge writes go through its doors, no push
    # lands on a merged PR).
    'DRAFT-UNTIL-READY': (TestPath('pyproject.toml'),),
    'FORGE-DOORS-ARE-CONSENT': (TestPath('pyproject.toml'),),
    # Adopted 2026-11-10 (dev-lab #320, as numerics-lab): the rendered `continue.yml` push job, which replaces the
    # retired NO-PUSH-ONTO-A-LANDED-PR, and the base pre-push row `closed-push`.
    'LANDED-BRANCH-CONTINUES': (TestPath('.github/workflows/continue.yml'),),
    'NO-PUSH-ONTO-A-CLOSED-PR': (TestPath('pyproject.toml'),),
    # Adopted 2026-11-07 with dev-lab #304: the base pre-push row `pr-current` this pyproject's hooks table runs.
    'PR-HEAD-IS-CURRENT': (TestPath('pyproject.toml'),),
    # Adopted 2026-11-13 with dev-lab 61560e40: the base pre-push row `milestone-base` (dev_lab.dev.milestonebase)
    # this pyproject's hooks table runs refuses both.
    'BASE-HOLDS-THE-MILESTONE': (TestPath('pyproject.toml'),),
    'VERSION-TAG-NEEDS-APPROVAL': (TestPath('pyproject.toml'),),
    # Adopted 2026-11-24 with dev-lab cbecf58a: the same milestonebase pre-push row refuses a non-version tag.
    'ONLY-VERSION-TAGS': (TestPath('pyproject.toml'),),
    # Adopted 2026-11-19 with the relock to dev-lab integrate/main: the base pre-push row `session-branch`
    # (dev_lab.dev.sessionbranch) this pyproject's hooks table runs refuses an off-task head.
    'PUSHED-HEAD-CARRIES-ITS-TASK': (TestPath('pyproject.toml'),),
    # Adopted 2026-11-19: the rendered settings run one `dev_lab.dev.hookdispatch` row per event, each with
    # timeout 5, and the dispatcher detaches a slow row.
    'HOOK-UNDER-FIVE-SECONDS': (TestPath('.claude/settings.json'),),
    # Adopted 2026-10-26 with dev-lab c9eaa7f: the agent guard ships in every family repo; the rendered
    # `.claude/hooks/deny-rules.json` refuses these, each by its own rule, and the test pins it live.
    **dict.fromkeys(
        ('AGENT-GUARD', 'SUBAGENT-NO-PUSH', 'WORKTREES-STAY-INSIDE', 'REPORTS-LIVE-IN-MEMORY',
         'INJECTED-TEXT-IS-PROGRESSIVE',
         # Gained 2026-10-29 with dev-lab 5abf8c4d: RECURSIVE-DELETE is a shipped deny rule, and the guard's
         # WIRING installs the `dev_lab.dev.pushwatch` PostToolUse advisory the live test pins.
         'RECURSIVE-DELETE', 'PUSHED-HEAD-IS-WATCHED', 'TASK-HEARS-ITS-FORGE-EVENTS',
         # Gained 2026-11-07 with dev-lab 8c4ab734: the guard refuses a write outside the checkout.
         'WRITES-STAY-IN-THE-TREE',
         # Gained 2026-11-13 with dev-lab 61560e40: the guard's WIRING installs the `dev_lab.dev.walls`
         # PostToolUse and Stop hooks the live test pins.
         'WALLS-ARE-RECORDED',
         # Gained 2026-11-17 with dev-lab b9875921: the per-module rows above became one `dev_lab.dev.hookdispatch`
         # row per event, and the deny engine took the journal door (TRACKED-CITES-NO-EPHEMERAL) and the
         # subagent refusal of heavy runs (HEAVY-IS-THE-MAIN-SESSIONS), each a shipped rule the live test pins.
         'TRACKED-CITES-NO-EPHEMERAL', 'HEAVY-IS-THE-MAIN-SESSIONS',
         # Gained 2026-11-22 with dev-lab 77ef8452: the rendered deny rows refuse code written under output/ and a
         # hash-like worktree name, and the same live wiring's Stop hook sends back a final message naming an
         # image by a relative path.
         'OUTPUT-HOLDS-NO-CODE', 'WORKTREE-NAMES-ITS-TASK', 'IMAGES-BY-ABSOLUTE-PATH'),
        (TestPath('tests/architecture/test_the_agent_guard_is_live.py'),),
    ),
    # Adopted 2026-11-20 with dev-lab 2adcab2: `[tool.dev_lab.dep] extras` is the one declaration; the rendered
    # workflows sync only it (ciworkflows refuses an `--all-extras` at render) and the render test pins them.
    'ALL-EXTRAS-REFUSED': (
        TestPath('pyproject.toml'), TestPath('tests/architecture/test_the_ci_workflow_is_rendered.py'),
    ),
    # Adopted 2026-11-26 with the dev-lab re-render: `[tool.dev_lab.env] entry` names ai_lab.hostenv and the
    # family scan pins every other tracked read; the secret-sink scan reads every tracked Python file; and the
    # forge's merge step reads `Red-On-Record` lines over the declared integrator table.
    'ENV-IS-READ-ONCE': (TestPath('tests/architecture/test_the_environment_is_read_once.py'),),
    'NEVER-LOG-SECRETS': (TestPath('tests/architecture/test_no_secret_is_logged.py'),),
    'RED-INTEGRATION-MERGES': (TestPath('pyproject.toml'),),
    # layout.toml declares the whole tree; the family scan refuses an undeclared directory or format, a second
    # config format and a backup copy under output/ or scratch/.
    **dict.fromkeys(('DECLARED-LAYOUT', 'NO-BACKUP-COPIES', 'ONE-FORMAT-PER-NEED'),
                    (TestPath('layout.toml'), TestPath('tests/architecture/test_the_tree_is_the_declared_layout.py'))),
}  # fmt: skip

#: Rules with no subject in this library today, BY NAME -- deriving it from ``RULES`` would silently
#: absorb a rule added upstream instead of making this repo decide about it. A rule leaves by gaining a
#: mechanism above; the set may only shrink.
_ABSENT: Final = frozenset({
    # Workflow machinery ai-lab does not run: no push gate, verdict ledger, forge door, lanes, sessions.
    'VERDICT-BAR-IS-THE-INCREMENT', 'NETWORK-RETRY-THEN-REPORT', 'SHARED-CHECKOUT', 'FORGE-THROUGH-THE-DOOR',
    'VERDICT-AS-STATUS', 'ISSUE-IS-INTENT', 'ONE-BRANCH-PER-SESSION', 'MERGE-DEVIATIONS-NAMED',
    'AUTO-MODE-RUNS-THE-DOORS', 'VERDICTS-FROM-THE-FORGE', 'MAIN-SESSION-PUBLISHES', 'ONE-BOX-ONE-LOCK',
    # The environment: no own venv door yet,
    # so nothing installs hooks or dependencies through a checked door here.
    'HOOKS-ARE-WIRED', 'INSTALL-DOOR-DELIVERS-THE-DECLARATION', 'ENV-MUTATION-THROUGH-THE-DOOR',
    'LATEST-DEPENDENCIES',
    # Domain and measurement rules: no physics, units, tolerances, bars, capability matrix or cases.
    'IMPLEMENT-EVERYTHING', 'BAR-IS-A-CONSTANT', 'UNITS-GO-THROUGH-PINT', 'TOLERANCE-CARRIES-A-UNIT',
    'REGISTRY-OWNS-THE-DECISION', 'PRODUCTION-ENTRY-POINT',
    # Docs and memory: no rules pages, injected docs or memory tree in this repo.
    'DOCS-SPLIT', 'MEMORY-SHAPE', 'INJECTED-DOC-WIDTH-CEILING',
    # Code placement and scratch: one src/ and one tests/ tree, no scratch.
    'CODE-IN-CODE-ROOTS', 'SCRATCH-IS-NEVER-COMMITTED', 'STRAY-FILE-REFUSED', 'SPLIT-BY-CONCEPT',
    # Meta-rules about guards: the guards above are the family's shared bodies, which carry them.
    'DECLARATION-LIES', 'PLANTED-CONTROL', 'FLOOR-ON-EVERY-SCAN', 'ESCAPE-HATCH-CEILING', 'RATCHET-TWO-SIDES',
    'NAMED-SETS-NOT-COUNTS', 'XFAIL-NOT-SKIP', 'MODULE-SIZE-ALARM',
    # Hosts: ai-lab deploys no host role, so it runs no scheduler to restrict, and declares no ssh endpoint
    # whose key it could check.
    'SCHEDULER-ONLY-FOR-SERVING', 'SSH-HOST-KEY-DECLARED', 'MACHINES-MEET-ONLY-THROUGH-THE-FORGE',
    # Opt-in architecture tables this library does not declare: no dependencies.toml, no [tool.dev_lab.layers]
    # or [tool.dev_lab.native], no crate to price, no recorded baselines.
    'OWNER-DAG', 'NO-DETOUR', 'ZERO-CORE-CHANGE', 'NATIVE-FIRST', 'MEASURE-BEFORE-PORTING',
    'BASELINE-REWRITE-QUOTED',
    # Guard-shipped rules whose hook rows this repo's agent guard does not render yet (no task ledger tracked,
    # no codeplace hook, no unit-parsing config reader of its own).
    'LEDGER-WRITES-THROUGH-ITS-DOOR', 'PRIOR-ART-FIRST', 'CONFIG-VALUES-CARRY-UNITS',
})  # fmt: skip

#: The size of the set as of 2026-10-08 (41), re-based 2026-10-13 when the registry added six rules with
#: no subject here (the five integration-branch rules and REPORTS-LIVE-IN-MEMORY), and again (48) for
#: PUSH-FROM-ITS-OWN-BRANCH, an integration-line rule with no line here; lowering it is the only
#: legal edit. Lowered to 47 on 2026-10-19: the registry retired INTEGRATION-LIVES-IN-THE-MAIN-CHECKOUT and
#: PUSH-FROM-ITS-OWN-BRANCH and added FORMATS-UPGRADE-ON-READ. Lowered to 43 the same day: the four
#: branch-flow rules gained mechanisms once integrate/main was declared. Lowered to 42 on 2026-10-20:
#: RETIRED-NAMES-REGISTERED gained ai_lab.retired and its scan. Lowered to 41 the same day:
#: FORMATS-UPGRADE-ON-READ gained the ledger's declared upgrade steps.
#: Raised to 44 on 2026-10-23: the dev-lab split brought three rules with no mechanism here yet
#: (STRAY-FILE-REFUSED, SPLIT-BY-CONCEPT, TRACKED-CITES-NO-EPHEMERAL).
#: Lowered to 39 on 2026-10-26: the agent guard took AGENT-GUARD and the four rules it refuses.
#: Lowered to 38 on 2026-11-07: NO-CJK-IN-TRACKED-SOURCE is always on (dev_lab.dev.famtests.alwayson).
#: Lowered to 37 on 2026-11-17: TRACKED-CITES-NO-EPHEMERAL became a shipped deny rule of the agent guard.
#: Raised to 38 on 2026-11-20: dev-lab 2adcab2 added SCHEDULER-ONLY-FOR-SERVING, a host-deploy rule with no
#: subject in this library.
#: Raised to 39 on 2026-11-22 (user ruling 2026-11-22): dev-lab 77ef8452 added SSH-HOST-KEY-DECLARED, and this
#: library declares no ssh endpoint.
#: Raised to 49 on 2026-11-26 (user ruling 2026-11-26: raise now, lower the hard ones in the full review): the
#: dev-lab re-render added sixteen rules; six gained mechanisms here, ten are declared absent above.
_ABSENT_CEILING: Final = 49


def test_every_rule_is_enforced_or_declared_absent() -> None:
    assert_adopted(_PROFILE, Adoption(app_name='ai_lab', mechanisms=_MECHANISMS, declared_absent=_ABSENT))


def test_the_absent_set_does_not_grow() -> None:
    assert len(_ABSENT) <= _ABSENT_CEILING, sorted(_ABSENT)


def test_no_rule_is_both_enforced_and_absent() -> None:
    assert not (_ABSENT & frozenset(_MECHANISMS))
