"""``python -m ai_lab.review (--range <a..b> | --bundle <dir>) --model <id> [--lens L] [--paths p ...]``: one review.

``--range`` reads the change with git in ``--repo``; ``--bundle`` reads a review bundle directory and
never starts git (:mod:`ai_lab.review.source`, chosen once here).

Prints the findings JSON (the method's ``findings-output.schema.json``) on stdout, exit 0, whatever the
reviewers' statuses. A refused invocation -- a method directory that is missing or of an unknown
``schema_version``, a range git cannot diff, a bundle that cannot be read, an unreadable ``--findings`` -- exits 2
naming it on stderr, with nothing on stdout.

A reviewer's prompt follows the change's SIZE TIER (user ruling 2026-10-31, :mod:`ai_lab.review.explore`): a
small change's whole diff, a medium one's CHANGE MAP (the changed files with their line counts and the directory
tree), a huge one's directory-level map and commit messages; in every tier it may ask each turn for the diffs,
files and greps it chooses. The forge reviews a
large head in SLICES (dev-lab ``forgereview``): each slice runs this with ``--paths`` (the map and every read
are restricted to them) and a ``--lens`` (``strict`` plans every eligible reviewer, ``standard`` the method's
count, ``light`` one). ``--findings <merged.json>`` is the SYNTHESIS pass: it reads the slices' merged findings
and the whole range's map, and reports what crosses slices.

The document opens with ``summary`` -- what the change does and why, derived each run from the range's
commits, merge resolutions, ``Merge-Audit`` lines, the memory it adds or changes and ``--pr-body``
(:mod:`ai_lab.review.summary`); ``--no-summary`` omits it (a slice the caller summarizes once). The summary is
written FIRST: one cut at its output budget exits 3 naming it on stderr, before any reviewer is paid for. ``--disposed
<file>`` (``{"disposed": [{id, reason}]}``) names ids disposed of earlier: they are not re-raised, and the
document lists them under ``disposed``.

THE MODEL CALLS ARE BATCHED BY DEFAULT (:mod:`ai_lab.batch`; ``--no-batch`` makes them at once): the summary is one
batch, then each turn of every reviewer one more. A run whose ``--wait`` is spent with a batch still running prints
nothing, says ``pending: <line>`` on stderr and exits :data:`~ai_lab.batch.PENDING_EXIT`; run again with the same
``--batch-state`` and it resumes that batch, never resubmitting. The reviewer plan is drawn from the state's seed, so
the replay plans the same reviewers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from ai_lab import batch
from ai_lab.call import answer
from ai_lab.client import catalogue
from ai_lab.review.explore import Changed, brief, change_map, commit_messages, serve, size, turns_for
from ai_lab.review.explore import diff as git_diff
from ai_lab.review.explore import numstat as git_numstat
from ai_lab.review.method import (
    METHOD_DIR,
    UnknownMethodVersion,
    configured_providers,
    load_method,
    principle_list,
    rule_ids,
)
from ai_lab.review.run import LENSES, run
from ai_lab.review.source import BundleSource, GitSource, Source
from ai_lab.review.summary import SummaryTruncated, change_summary, git_intent
from ai_lab.review.tiers import evidence, ledger_path, record

__all__ = []

_EXIT_REFUSED: Final = 2
_EXIT_TRUNCATED: Final = 3

#: The synthesis pass's change text: the slices' findings, then the range's file summary.
_SYNTHESIS: Final = (
    'SYNTHESIS PASS. This change was reviewed in slices; the findings below are already reported. '
    'Do not repeat them: report only what crosses slices (an interface changed on one side and not the other, '
    'a duplicated fact, a contradiction between parts).\n\nFindings already reported:\n{findings}\n\n'
    'Files changed in the range:\n{stat}'
)


def _change(args: argparse.Namespace, source: Source, rows: Sequence[Changed]) -> dict:
    """What the reviewers start from: the slice's tier brief, or for ``--findings`` the synthesis brief (explored)."""
    if args.findings is None:
        return brief(source.label, rows, read_diff=lambda: source.diff(args.paths), read_log=source.log)
    findings = json.loads(Path(args.findings).read_text(encoding='utf-8'))
    reported = json.dumps(findings, ensure_ascii=False, indent=1)
    text = _SYNTHESIS.format(findings=reported, stat=change_map(source.label, rows))
    return {'mode': 'medium', 'manifest': text, 'diff': '', 'commits': '', 'diff_bytes': size(text)}


def _disposed(path: str | None) -> dict[str, str]:
    """The ``{id: reason}`` the ``--disposed`` file names; ``ValueError``/``TypeError`` when it is not that shape."""
    if path is None:
        return {}
    rows = json.loads(Path(path).read_text(encoding='utf-8')).get('disposed')
    if not isinstance(rows, list) or not all(isinstance(row, dict) and {'id', 'reason'} <= set(row) for row in rows):
        msg = f'{path} is not {{"disposed": [{{id, reason}}]}}'
        raise ValueError(msg)
    return {str(row['id']): str(row['reason']) for row in rows}


def _summary(args: argparse.Namespace, source: Source, call: Callable, provider: str) -> dict | None:
    """The summary section, ``None`` when the source reads no intent.

    An intent the source cannot read is the section's ``error``, never a refused review.
    """
    try:
        found = source.intent()
    except ValueError as unread:
        found, error = {'commits': [], 'resolutions': [], 'merge_audit': [], 'memory': []}, str(unread)
    else:
        error = None
    if found is None:
        return None
    found['pr_body'] = Path(args.pr_body).read_text(encoding='utf-8') if args.pr_body else ''
    summary = change_summary(found, call=call, provider=provider, model=args.model)
    if error is not None:
        summary['error'] = error
    return summary


def _parser() -> argparse.ArgumentParser:
    """The command line's flags, the batch runner's among them."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.review', description=__doc__.split('\n', 1)[0])
    parser.add_argument(
        '--method', default=METHOD_DIR, help='a method directory read in place (default: the packaged one)'
    )
    given = parser.add_mutually_exclusive_group(required=True)
    given.add_argument('--range', dest='range_', help='the git range to review, e.g. main...HEAD')
    given.add_argument('--bundle', help='a review bundle directory to review instead: no git is started')
    parser.add_argument('--model', required=True, help='the model id for every provider the method names none for')
    parser.add_argument('--providers', nargs='+', help="override the default config's providers.headless")
    parser.add_argument('--repo', default='.', help='the repository the range is in (default: the working directory)')
    parser.add_argument('--lens', choices=sorted(LENSES), default='standard', help='how many reviewers to plan')
    parser.add_argument(
        '--paths', nargs='*', default=[], help='the slice: map only these paths (none: the whole range)'
    )
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
    batch.add_arguments(parser)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    call: Callable[[Mapping], Mapping] = answer,
    numstat: Callable[..., list[Changed]] = git_numstat,
    diff: Callable[..., str] = git_diff,
    log: Callable[[str, str], str] = commit_messages,
    intent: Callable[[str, str], dict] | None = git_intent,
    connect: Callable[..., object] = batch.connect,
) -> int:
    """Run one headless review and print its findings JSON; exit 2 on a refused invocation."""
    args = _parser().parse_args(argv)
    try:
        method = load_method(args.method, principles=principle_list(args.principles), registry=rule_ids(args.rules))
        source: Source = (
            BundleSource(Path(args.bundle)) if args.bundle else
            GitSource(args.range_, args.repo, read_numstat=numstat, read_diff=diff, read_log=log, read_intent=intent)
        )  # fmt: skip
        rows = source.numstat(() if args.findings else args.paths)
        change = _change(args, source, rows)
        disposed = _disposed(args.disposed)
        runs = batch.from_arguments(args, answer=call, connect=connect)
    except (FileNotFoundError, UnknownMethodVersion, ValueError, TypeError) as refused:
        sys.stderr.write(f'refused: {refused}\n')
        return _EXIT_REFUSED

    provider = (args.providers or configured_providers({}, 'headless')
                or method.spec['run_modes']['headless']['providers'])[0]  # fmt: skip
    label = f'review:{source.label}:{",".join(args.paths)}:{args.lens}:{bool(args.findings)}'

    def reviewed(call: Callable[[Mapping], Mapping]) -> tuple[dict | None, dict, dict]:
        """One pass: the summary (alone, before any reviewer is paid for), then every reviewer."""
        spent: dict = {'turns': 0, 'read_bytes': 0, 'usage': {}}

        def explore(asked: Mapping) -> str:
            text = serve(asked, source=source, rows=rows)
            spent['read_bytes'] += size(text)
            return text

        def counted(request: Mapping) -> Mapping:
            reply = call(request)
            spent['turns'] += 1
            for key, value in (reply.get('usage') or {}).items():
                if isinstance(value, int):
                    spent['usage'][key] = spent['usage'].get(key, 0) + value
            return reply

        summary = None if args.no_summary else _summary(args, source, call, provider)
        doc = run(
            method, range_=source.label, brief=change, explore=explore, model=args.model, providers=args.providers,
            available=set(catalogue()), call=counted, rng=runs.rng(label), lens=args.lens, disposed=disposed,
            turns=turns_for(change['mode']),
        )  # fmt: skip
        return summary, doc, spent

    started = time.monotonic()
    try:
        settled = runs.settle(reviewed)
    except SummaryTruncated as cut:
        sys.stderr.write(f'truncated: {cut}\n')
        return _EXIT_TRUNCATED
    if isinstance(settled, batch.Pending):
        sys.stderr.write(f'pending: {settled.line()}\n')
        return batch.PENDING_EXIT
    summary, doc, spent = settled
    identity = source.identity()
    record(ledger_path(identity['repo']), evidence(
        label=source.label, identity=identity, rows=rows, tier='synthesis' if args.findings else change['mode'],
        diff_bytes=change['diff_bytes'], turns=spent['turns'], read_bytes=spent['read_bytes'],
        findings=doc['findings'], seconds=time.monotonic() - started, usage=spent['usage'],
    ))  # fmt: skip
    if summary is not None:
        doc = {'summary': summary, **doc}
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
