"""``python -m ai_lab.review --method <dir> --range <a..b> --model <id> [--providers p ...]``: one headless review.

Prints the findings JSON (the method's ``findings-output.schema.json``) on stdout, exit 0, whatever the
reviewers' statuses. A refused invocation -- a method directory that is missing or of an unknown
``schema_version``, a range git cannot diff -- exits 2 naming it on stderr, with nothing on stdout.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Final

from ai_lab.call import answer
from ai_lab.client import catalogue
from ai_lab.review.method import UnknownMethodVersion, load_method
from ai_lab.review.run import run

__all__ = []

_EXIT_REFUSED: Final = 2


def _git_diff(range_: str, repo: str) -> str:
    git = shutil.which('git')
    if git is None:
        msg = 'git is not on PATH: a review reads its diff with git'
        raise ValueError(msg)
    done = subprocess.run(
        [git, '-C', repo, 'diff', '--end-of-options', range_],
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=False,
    )
    if done.returncode:
        msg = f'git diff {range_} in {repo} failed: {done.stderr.strip()}'
        raise ValueError(msg)
    return done.stdout


def main(
    argv: Sequence[str] | None = None,
    *,
    call: Callable[[Mapping], Mapping] = answer,
    diff: Callable[[str, str], str] = _git_diff,
) -> int:
    """Run one headless review and print its findings JSON; exit 2 on a refused invocation."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.review', description=__doc__.split('\n', 1)[0])
    parser.add_argument('--method', required=True, help="sharp-review's method directory (read in place)")
    parser.add_argument('--range', required=True, dest='range_', help='the git range to review, e.g. main...HEAD')
    parser.add_argument('--model', required=True, help='the model id for every provider the method names none for')
    parser.add_argument('--providers', nargs='+', help="override run_modes.headless.providers (default ['anthropic'])")
    parser.add_argument('--repo', default='.', help='the repository the range is in (default: the working directory)')
    args = parser.parse_args(argv)
    try:
        method = load_method(args.method)
        text = diff(args.range_, args.repo)
    except (FileNotFoundError, UnknownMethodVersion, ValueError) as refused:
        sys.stderr.write(f'refused: {refused}\n')
        return _EXIT_REFUSED
    doc = run(
        method, range_=args.range_, diff=text, model=args.model, providers=args.providers,
        available=set(catalogue()), call=call, rng=random.SystemRandom(), date=datetime.now(UTC).date().isoformat(),
    )  # fmt: skip
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
