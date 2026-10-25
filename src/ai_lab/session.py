"""``ai-lab-session <claude|codex> [args]``: mint the session id, then run the agent CLI with ``LAB_SESSION`` set.

The ONE launcher every launch path calls (PLAN-a-short-robust-toolchain P4). ``LAB_SESSION`` is one id per agent
session, inherited by every child (a subagent, a takeover, a codex run) and stamped as ``Lab-Session`` at
commit-msg by dev-lab. The id is, in order:

1. inherited -- ``LAB_SESSION``, or the runtime's own (``CLAUDE_CODE_SESSION_ID``, ``CODEX_SESSION_ID``,
   ``CODEX_THREAD_ID``): a launch inside a session belongs to it;
2. the vendor's conversation id named in the args (``--resume <id>``, ``-r <id>``, ``--session-id <id>``;
   ``codex resume <id>``), so a resumed conversation keeps its id;
3. a fresh UUID. A fresh Claude conversation is pinned to it with ``--session-id`` so a later ``--resume <id>``
   finds the same id; one that cannot be pinned (``--continue``, a bare ``--resume`` picker, a ``claude
   <command>``) or a fresh Codex launch only carries it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Final

#: ``main`` is reached through the console script, not imported, so it is not declared here.
__all__ = ['AGENTS', 'ENV', 'launch', 'session_id']

#: The environment variable the session id is issued in.
ENV: Final = 'LAB_SESSION'

#: The agent CLIs this launcher runs.
AGENTS: Final = ('claude', 'codex')

#: The runtimes' own session ids, read after :data:`ENV`.
_NATIVE: Final = ('CLAUDE_CODE_SESSION_ID', 'CODEX_SESSION_ID', 'CODEX_THREAD_ID')

#: ``claude <command>`` runs no conversation, so it is never pinned (``claude --help``, 2026-10-21).
_CLAUDE_COMMANDS: Final = frozenset({
    'agents', 'attach', 'auth', 'auto-mode', 'doctor', 'gateway', 'import', 'install', 'logs', 'mcp', 'plugin',
    'plugins', 'purge', 'respawn', 'rm', 'setup-token', 'stop', 'kill', 'ultrareview', 'update', 'upgrade',
})  # fmt: skip
_UNPINNABLE: Final = frozenset({'--continue', '-c', '--resume', '-r', '--session-id'})
_EXIT_USAGE: Final = 2


def _value(args: Sequence[str], names: Sequence[str]) -> str | None:
    for i, arg in enumerate(args):
        for name in names:
            if arg == name and i + 1 < len(args) and not args[i + 1].startswith('-'):
                return args[i + 1]
            if arg.startswith(f'{name}='):
                return arg[len(name) + 1 :] or None
    return None


def _vendor(agent: str, args: Sequence[str]) -> str | None:
    """The conversation id *args* name for *agent*, or ``None``."""
    if agent == 'codex':
        return args[1] if len(args) > 1 and args[0] == 'resume' and not args[1].startswith('-') else None
    return _value(args, ['--session-id']) or _value(args, ['--resume', '-r'])


def session_id(env: Mapping[str, str], vendor: str | None = None) -> str:
    """The inherited id, else *vendor*'s, else a fresh UUID."""
    inherited = next((env[key] for key in (ENV, *_NATIVE) if env.get(key)), None)
    return inherited or vendor or str(uuid.uuid4())


def launch(agent: str, args: Sequence[str], env: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """The environment and argv to run *agent* with: :data:`ENV` set, a fresh Claude conversation pinned."""
    vendor = _vendor(agent, args)
    inherited = any(env.get(key) for key in (ENV, *_NATIVE))
    session = session_id(env, vendor)
    argv = [agent, *args]
    fresh = not inherited and vendor is None
    pinnable = agent == 'claude' and not (args and args[0] in _CLAUDE_COMMANDS) and not _UNPINNABLE & set(args)
    if fresh and pinnable:
        argv += ['--session-id', session]
    return {**env, ENV: session}, argv


def _run(argv: Sequence[str], env: Mapping[str, str]) -> int:
    """Run *argv* with *env*, the console attached; its exit code (a subprocess: ``exec`` is not portable)."""
    found = shutil.which(argv[0])
    if found is None:
        sys.stderr.write(f'ai-lab-session: {argv[0]!r} is not installed on this host\n')
        return 127
    return subprocess.run([found, *argv[1:]], env=dict(env), check=False).returncode


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    runner: Callable[[Sequence[str], Mapping[str, str]], int] = _run,
) -> int:
    """Launch the named agent with its session; exit 2 naming :data:`AGENTS` otherwise."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in AGENTS:
        got = repr(args[0]) if args else 'no agent'
        sys.stderr.write(f'ai-lab-session: {got}; the agents are {list(AGENTS)}\n')
        raise SystemExit(_EXIT_USAGE)
    child_env, child_argv = launch(args[0], args[1:], os.environ if env is None else env)
    return runner(child_argv, child_env)


if __name__ == '__main__':
    raise SystemExit(main())
