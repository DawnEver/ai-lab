"""``python -m ai_lab.review [--method <dir>] --range <a..b> --model <id> [--lens L] [--paths p ...]``: one review.

Prints the findings JSON (the method's ``findings-output.schema.json``) on stdout, exit 0, whatever the
reviewers' statuses. A refused invocation -- a method directory that is missing or of an unknown
``schema_version``, a range git cannot diff, an unreadable ``--findings`` -- exits 2 naming it on stderr,
with nothing on stdout.

The forge reviews a large head in SLICES (dev-lab ``forgereview``): each slice runs this with ``--paths``
(only those paths' diff is read) and a ``--lens`` (``strict`` plans every eligible reviewer, ``standard``
the method's count, ``light`` one). ``--findings <merged.json>`` is the SYNTHESIS pass: it reads the slices'
merged findings and the range's file summary -- never the whole diff, which is what did not fit -- and
reports what crosses slices.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from ai_lab.call import answer
from ai_lab.client import catalogue
from ai_lab.review.method import METHOD_DIR, UnknownMethodVersion, load_method, principle_list, rule_ids
from ai_lab.review.run import LENSES, run

__all__ = []

_EXIT_REFUSED: Final = 2

#: The synthesis pass's change text: the slices' findings, then the range's file summary.
_SYNTHESIS: Final = (
    'SYNTHESIS PASS. This change was reviewed in slices; the findings below are already reported. '
    'Do not repeat them: report only what crosses slices (an interface changed on one side and not the other, '
    'a duplicated fact, a contradiction between parts).\n\nFindings already reported:\n{findings}\n\n'
    'Files changed in the range:\n{stat}'
)


def _git_diff(range_: str, repo: str, paths: Sequence[str] = (), *, stat: bool = False) -> str:
    """The diff of *range_* (its file summary with *stat*), restricted to *paths* when any are named."""
    git = shutil.which('git')
    if git is None:
        msg = 'git is not on PATH: a review reads its diff with git'
        raise ValueError(msg)
    argv = [git, '-C', repo, 'diff', *(['--stat'] if stat else []), '--end-of-options', range_]
    done = subprocess.run(
        [*argv, *(['--', *paths] if paths else [])],
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=False,
    )
    if done.returncode:
        msg = f'git diff {range_} in {repo} failed: {done.stderr.strip()}'
        raise ValueError(msg)
    return done.stdout


def _git_stat(range_: str, repo: str) -> str:
    return _git_diff(range_, repo, stat=True)


def _change(args: argparse.Namespace, diff: Callable[..., str], stat: Callable[[str, str], str]) -> str:
    """The text the reviewers read: the slice's diff, or for ``--findings`` the synthesis brief."""
    if args.findings is None:
        return diff(args.range_, args.repo, tuple(args.paths))
    findings = json.loads(Path(args.findings).read_text(encoding='utf-8'))
    reported = json.dumps(findings, ensure_ascii=False, indent=1)
    return _SYNTHESIS.format(findings=reported, stat=stat(args.range_, args.repo))


def main(
    argv: Sequence[str] | None = None,
    *,
    call: Callable[[Mapping], Mapping] = answer,
    diff: Callable[..., str] = _git_diff,
    stat: Callable[[str, str], str] = _git_stat,
) -> int:
    """Run one headless review and print its findings JSON; exit 2 on a refused invocation."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.review', description=__doc__.split('\n', 1)[0])
    parser.add_argument(
        '--method', default=METHOD_DIR, help='a method directory read in place (default: the packaged one)'
    )
    parser.add_argument('--range', required=True, dest='range_', help='the git range to review, e.g. main...HEAD')
    parser.add_argument('--model', required=True, help='the model id for every provider the method names none for')
    parser.add_argument('--providers', nargs='+', help="override the default config's providers.headless")
    parser.add_argument('--repo', default='.', help='the repository the range is in (default: the working directory)')
    parser.add_argument('--lens', choices=sorted(LENSES), default='standard', help='how many reviewers to plan')
    parser.add_argument('--paths', nargs='+', default=[], help="the slice: read only these paths' diff")
    parser.add_argument('--findings', help="synthesis: the slices' merged findings JSON (the whole diff is not read)")
    parser.add_argument(
        '--rules', help='the registry rule ids, comma-separated (required: the caller owns the registry)'
    )
    parser.add_argument(
        '--principles', help='a JSON file {"principles": [{id, name, statement}]} (required: the caller owns them)'
    )
    args = parser.parse_args(argv)
    try:
        method = load_method(args.method, principles=principle_list(args.principles), registry=rule_ids(args.rules))
        text = _change(args, diff, stat)
    except (FileNotFoundError, UnknownMethodVersion, ValueError, TypeError) as refused:
        sys.stderr.write(f'refused: {refused}\n')
        return _EXIT_REFUSED
    doc = run(
        method, range_=args.range_, diff=text, model=args.model, providers=args.providers,
        available=set(catalogue()), call=call, rng=random.SystemRandom(), lens=args.lens,
    )  # fmt: skip
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
