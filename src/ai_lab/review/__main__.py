"""``python -m ai_lab.review [--method <dir>] --range <a..b> --model <id> [--lens L] [--paths p ...]``: one review.

Prints the findings JSON (the method's ``findings-output.schema.json``) on stdout, exit 0, whatever the
reviewers' statuses. A refused invocation -- a method directory that is missing or of an unknown
``schema_version``, a range git cannot diff, an unreadable ``--findings`` -- exits 2 naming it on stderr,
with nothing on stdout.

A reviewer is never handed the diff (user ruling 2026-10-31): it gets the range's CHANGE MAP -- the changed
files with their line counts and the directory tree, the directory level alone past a bound -- and explores,
asking each turn for the diffs, files and greps it chooses (:mod:`ai_lab.review.explore`). The forge reviews a
large head in SLICES (dev-lab ``forgereview``): each slice runs this with ``--paths`` (the map and every read
are restricted to them) and a ``--lens`` (``strict`` plans every eligible reviewer, ``standard`` the method's
count, ``light`` one). ``--findings <merged.json>`` is the SYNTHESIS pass: it reads the slices' merged findings
and the whole range's map, and reports what crosses slices.

The document opens with ``summary`` -- what the change does and why, derived each run from the range's
commits, merge resolutions, ``Merge-Audit`` lines, the memory it adds or changes and ``--pr-body``
(:mod:`ai_lab.review.summary`); ``--no-summary`` omits it (a slice the caller summarizes once). ``--disposed
<file>`` (``{"disposed": [{id, reason}]}``) names ids disposed of earlier: they are not re-raised, and the
document lists them under ``disposed``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from ai_lab.call import answer
from ai_lab.client import catalogue
from ai_lab.review.explore import Changed, change_map, serve
from ai_lab.review.explore import numstat as git_numstat
from ai_lab.review.method import METHOD_DIR, UnknownMethodVersion, load_method, principle_list, rule_ids
from ai_lab.review.run import LENSES, run
from ai_lab.review.summary import change_summary, git_intent

__all__ = []

_EXIT_REFUSED: Final = 2

#: The synthesis pass's change text: the slices' findings, then the range's file summary.
_SYNTHESIS: Final = (
    'SYNTHESIS PASS. This change was reviewed in slices; the findings below are already reported. '
    'Do not repeat them: report only what crosses slices (an interface changed on one side and not the other, '
    'a duplicated fact, a contradiction between parts).\n\nFindings already reported:\n{findings}\n\n'
    'Files changed in the range:\n{stat}'
)


def _change(args: argparse.Namespace, rows: Sequence[Changed]) -> str:
    """The text the reviewers start from: the slice's change map, or for ``--findings`` the synthesis brief."""
    if args.findings is None:
        return change_map(args.range_, rows)
    findings = json.loads(Path(args.findings).read_text(encoding='utf-8'))
    reported = json.dumps(findings, ensure_ascii=False, indent=1)
    return _SYNTHESIS.format(findings=reported, stat=change_map(args.range_, rows))


def _disposed(path: str | None) -> dict[str, str]:
    """The ``{id: reason}`` the ``--disposed`` file names; ``ValueError``/``TypeError`` when it is not that shape."""
    if path is None:
        return {}
    rows = json.loads(Path(path).read_text(encoding='utf-8')).get('disposed')
    if not isinstance(rows, list) or not all(isinstance(row, dict) and {'id', 'reason'} <= set(row) for row in rows):
        msg = f'{path} is not {{"disposed": [{{id, reason}}]}}'
        raise ValueError(msg)
    return {str(row['id']): str(row['reason']) for row in rows}


def _summary(args: argparse.Namespace, intent: Callable[[str, str], dict], call: Callable, provider: str) -> dict:
    """The summary section; an intent git cannot read is the section's ``error``, never a refused review."""
    try:
        found = intent(args.range_, args.repo)
    except ValueError as unread:
        found, error = {'commits': [], 'resolutions': [], 'merge_audit': [], 'memory': []}, str(unread)
    else:
        error = None
    found['pr_body'] = Path(args.pr_body).read_text(encoding='utf-8') if args.pr_body else ''
    summary = change_summary(found, call=call, provider=provider, model=args.model)
    if error is not None:
        summary['error'] = error
    return summary


def main(
    argv: Sequence[str] | None = None,
    *,
    call: Callable[[Mapping], Mapping] = answer,
    numstat: Callable[..., list[Changed]] = git_numstat,
    intent: Callable[[str, str], dict] | None = git_intent,
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
    parser.add_argument('--paths', nargs='+', default=[], help='the slice: map and read only these paths')
    parser.add_argument('--findings', help="synthesis: the slices' merged findings JSON, read with the range's map")
    parser.add_argument(
        '--rules', help='the registry rule ids, comma-separated (required: the caller owns the registry)'
    )
    parser.add_argument(
        '--principles', help='a JSON file {"principles": [{id, name, statement}]} (required: the caller owns them)'
    )
    parser.add_argument('--pr-body', help='a file holding the PR body, read for the summary')
    parser.add_argument('--no-summary', action='store_true', help='omit the summary section (and its call)')
    parser.add_argument('--disposed', help='a JSON file {"disposed": [{id, reason}]}: ids not to re-raise')
    args = parser.parse_args(argv)
    try:
        method = load_method(args.method, principles=principle_list(args.principles), registry=rule_ids(args.rules))
        rows = numstat(args.range_, args.repo, () if args.findings else tuple(args.paths))
        text = _change(args, rows)
        disposed = _disposed(args.disposed)
    except (FileNotFoundError, UnknownMethodVersion, ValueError, TypeError) as refused:
        sys.stderr.write(f'refused: {refused}\n')
        return _EXIT_REFUSED

    def explore(asked: Mapping) -> str:
        return serve(asked, range_=args.range_, repo=args.repo, rows=rows)

    doc = run(
        method, range_=args.range_, change=text, explore=explore, model=args.model, providers=args.providers,
        available=set(catalogue()), call=call, rng=random.SystemRandom(), lens=args.lens, disposed=disposed,
    )  # fmt: skip
    if not args.no_summary and intent is not None:
        provider = doc['reviewers'][0]['provider'] if doc['reviewers'] else 'anthropic'
        doc = {'summary': _summary(args, intent, call, provider), **doc}
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
