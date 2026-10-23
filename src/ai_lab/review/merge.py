"""``python -m ai_lab.review.merge --method <dir>``: the merge of one recorded run, read on stdin.

The run record is ``{range, date, profiles, reviewers, results}``, the shape sharp-review's planned
fan-out records (``results`` absent means no reviewer returned). It is merged by the method's rules --
the ONE merge every sharp-review executor uses -- and the findings JSON (the method's
``findings-output.schema.json``) printed on stdout, exit 0. A refused invocation -- a method directory
that is missing or of an unknown ``schema_version``, a record that is not a run record -- exits 2 with
``refused: <why>`` on stderr and nothing on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Final

from ai_lab.review.method import UnknownMethodVersion, load_method
from ai_lab.review.rules import merge

__all__ = []

_EXIT_REFUSED: Final = 2


def _run_record(text: str) -> dict:
    """The run record *text* holds, as ``merge``'s keywords; ``ValueError``/``TypeError`` naming what is wrong."""
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


def main(argv: Sequence[str] | None = None) -> int:
    """Merge the run record on stdin and print its findings JSON; exit 2 when refused."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.review.merge', description=__doc__.split('\n', 1)[0])
    parser.add_argument('--method', required=True, help="sharp-review's method directory (read in place)")
    args = parser.parse_args(argv)
    try:
        doc = merge(load_method(args.method), **_run_record(sys.stdin.read()))
    except (FileNotFoundError, UnknownMethodVersion, ValueError, TypeError) as refused:
        sys.stderr.write(f'refused: {refused}\n')
        return _EXIT_REFUSED
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
