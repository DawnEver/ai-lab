"""``ai-lab-session <claude|codex> [args]``: the ONE place a session id is minted and the agent CLI launched.

The id is inherited when a parent session already has one, else the vendor's own conversation id named in the
args (so a resume keeps it), else a fresh UUID; a fresh Claude conversation is pinned to it with
``--session-id`` so a later ``--resume <id>`` finds the same id again.
"""

import uuid

import pytest

from ai_lab.session import AGENTS, ENV, launch, main, session_id

_FRESH = '0b5c3a52-1f7e-4c3e-9a77-5d1e2f3a4b6c'


def test_an_inherited_session_is_kept_and_never_re_pinned() -> None:
    env, argv = launch('claude', ['-p', 'hi'], {ENV: 'parent-session-1'})
    assert env[ENV] == 'parent-session-1'
    assert argv == ['claude', '-p', 'hi']


def test_a_fresh_claude_launch_mints_a_uuid_and_pins_the_conversation_to_it() -> None:
    env, argv = launch('claude', ['-p', 'hi'], {})
    uuid.UUID(env[ENV])
    assert argv == ['claude', '-p', 'hi', '--session-id', env[ENV]]


@pytest.mark.parametrize(
    ('args', 'expected'),
    [(['--resume', _FRESH], _FRESH), (['-r', _FRESH], _FRESH), ([f'--session-id={_FRESH}'], _FRESH)],
    ids=['resume', 'r', 'session-id'],
)
def test_a_named_claude_conversation_is_the_session(args: list[str], expected: str) -> None:
    env, argv = launch('claude', args, {})
    assert env[ENV] == expected
    assert argv == ['claude', *args]


@pytest.mark.parametrize('args', [['--continue'], ['-c'], ['--resume'], ['mcp', 'list']])
def test_a_claude_launch_that_cannot_be_pinned_gets_a_fresh_id_and_no_pin(args: list[str]) -> None:
    """PLANTED control beside the fresh launch: the same args without these are pinned."""
    env, argv = launch('claude', args, {})
    uuid.UUID(env[ENV])
    assert argv == ['claude', *args]


def test_a_codex_resume_keeps_the_thread_id_and_a_fresh_codex_launch_is_never_pinned() -> None:
    env, argv = launch('codex', ['resume', _FRESH], {})
    assert (env[ENV], argv) == (_FRESH, ['codex', 'resume', _FRESH])
    env, argv = launch('codex', ['exec', 'x'], {})
    uuid.UUID(env[ENV])
    assert argv == ['codex', 'exec', 'x']


def test_a_native_runtime_id_wins_over_a_fresh_uuid() -> None:
    assert session_id({'CLAUDE_CODE_SESSION_ID': 'native-claude-1'}) == 'native-claude-1'
    assert session_id({'CODEX_THREAD_ID': 'native-codex-1'}) == 'native-codex-1'
    assert session_id({}, vendor='vendor-id-1') == 'vendor-id-1'


def test_main_runs_the_agent_with_the_session_set_and_returns_its_exit() -> None:
    ran = []

    def runner(argv, env):
        ran.append((argv, env[ENV]))
        return 7

    assert main(['codex', 'exec', 'x'], env={ENV: 'parent-session-1'}, runner=runner) == 7
    assert ran == [(['codex', 'exec', 'x'], 'parent-session-1')]


def test_an_unknown_agent_exits_2_naming_the_known_ones(capsys) -> None:
    with pytest.raises(SystemExit) as refused:
        main(['gemini'], env={}, runner=lambda *_: 0)
    assert refused.value.code == 2
    err = capsys.readouterr().err
    assert all(agent in err for agent in AGENTS)
