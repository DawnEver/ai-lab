"""``ai-lab <command> [args]``: the machine install's one entry point.

A host runs ``ai-lab review ...`` from the installed command, so its configuration names no interpreter
path (PLAN-a-short-robust-toolchain P4). Each command is the module's own ``main``; its arguments and exit
code pass through unchanged.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Final

from ai_lab.review.__main__ import main as _review

#: ``main`` is reached through the ``ai-lab`` console script, not imported, so it is not re-declared here.
__all__ = ['COMMANDS']

#: Command name -> its ``main(argv) -> exit code``.
COMMANDS: Final[dict[str, Callable[[Sequence[str]], int]]] = {'review': _review}

_EXIT_USAGE: Final = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the named command with the remaining arguments; exit 2 naming the known commands otherwise."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in COMMANDS:
        got = repr(args[0]) if args else 'no command'
        sys.stderr.write(f'ai-lab: {got}; the commands are {sorted(COMMANDS)}\n')
        raise SystemExit(_EXIT_USAGE)
    return COMMANDS[args[0]](args[1:])
