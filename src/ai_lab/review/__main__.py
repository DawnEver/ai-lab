"""``python -m ai_lab.review run|merge --method <dir> ...``: a headless review, or the merge of a recorded run.

``run --range <a..b> --model <id> [--providers p ...]`` reviews a git range through ``respond``.
``merge`` reads one run record on stdin -- ``{range, date, profiles, reviewers, results}``, the shape
sharp-review's planned fan-out records (``results`` absent means no reviewer returned) -- and merges it by
the method's rules: the ONE merge every sharp-review executor uses.

Both print the findings JSON (the method's ``findings-output.schema.json``) on stdout, exit 0, whatever the
reviewers' statuses. A refused invocation -- a method directory that is missing or of an unknown
``schema_version``, a range git cannot diff, a run record that is not one -- exits 2 naming it on stderr,
with nothing on stdout.
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
from ai_lab.review.rules import merge
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


def _run_record(text: str) -> dict:
    """The run record *text* holds, as ``merge``'s keywords; ``ValueError`` naming what is wrong."""
    try:
        record = json.loads(text)
    except json.JSONDecodeError as bad:
        msg = f'the run record on stdin is not JSON: {bad}'
        raise ValueError(msg) from None
    if not isinstance(record, dict):
        msg = f'the run record on stdin is a {type(record).__name__}, not an object'
        raise TypeError(msg)
    missing = [k for k in ('range', 'date', 'profiles', 'reviewers') if k not in record]
    if missing:
        msg = f'the run record on stdin lacks {missing}'
        raise ValueError(msg)
    return {
        'range_': record['range'], 'date': record['date'], 'profiles': record['profiles'],
        'reviewers': record['reviewers'], 'results': record.get('results') or [],
    }  # fmt: skip


def main(
    argv: Sequence[str] | None = None,
    *,
    call: Callable[[Mapping], Mapping] = answer,
    diff: Callable[[str, str], str] = _git_diff,
) -> int:
    """Run one headless review, or merge one recorded run, and print its findings JSON; exit 2 when refused."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.review', description=__doc__.split('\n', 1)[0])
    verbs = parser.add_subparsers(dest='verb', required=True)
    review = verbs.add_parser('run', help='review a git range through respond')
    review.add_argument('--method', required=True, help="sharp-review's method directory (read in place)")
    review.add_argument('--range', required=True, dest='range_', help='the git range to review, e.g. main...HEAD')
    review.add_argument('--model', required=True, help='the model id for every provider the method names none for')
    review.add_argument('--providers', nargs='+', help="override run_modes.headless.providers (default ['anthropic'])")
    review.add_argument('--repo', default='.', help='the repository the range is in (default: the working directory)')
    merging = verbs.add_parser('merge', help='merge the run record on stdin by the method')
    merging.add_argument('--method', required=True, help="sharp-review's method directory (read in place)")
    args = parser.parse_args(argv)
    try:
        method = load_method(args.method)
        if args.verb == 'merge':
            doc = merge(method, **_run_record(sys.stdin.read()))
        else:
            text = diff(args.range_, args.repo)
    except (FileNotFoundError, UnknownMethodVersion, ValueError, TypeError) as refused:
        sys.stderr.write(f'refused: {refused}\n')
        return _EXIT_REFUSED
    if args.verb == 'run':
        doc = run(
            method, range_=args.range_, diff=text, model=args.model, providers=args.providers,
            available=set(catalogue()), call=call, rng=random.SystemRandom(), date=datetime.now(UTC).date().isoformat(),
        )  # fmt: skip
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
