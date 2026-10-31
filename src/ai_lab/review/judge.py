"""``python -m ai_lab.review.judge --range <a..b> --model <id> --rules <ids> --principles <f> --findings <f>``.

RE-JUDGE (user ruling 2026-10-31): every review round judges each open finding again against the new head, and an
author's claim that a finding is not a defect is adjudicated by an independent reviewer, never taken on trust.

``--findings`` is ``{"findings": [{id, severity, security, rule, file, anchor, summary, detail, suggestion,
claim}]}``. Each finding gets ONE judge call through the review's bounded explore loop
(:func:`ai_lab.review.run.explored`), starting from the range's change map (never its diff). stdout is one JSON
line ``{"schema_version": 1, "verdicts": [{id, outcome, evidence}]}``, one verdict per finding in input order, exit
0. A failed call or a reply with no valid outcome is ``present`` with evidence ``not judged: <why>`` -- it fails
closed, so the finding stays open. A refused invocation (a missing or bad file, an unknown method version, a range
git cannot read) exits 2 naming it on stderr, with nothing on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from ai_lab.call import answer
from ai_lab.review.explore import TURNS, change_map, numstat, serve
from ai_lab.review.method import (
    METHOD_DIR,
    UnknownMethodVersion,
    configured_providers,
    load_method,
    principle_list,
    rule_ids,
)
from ai_lab.review.run import REVIEW_TASK, explored

__all__ = ['OUTCOMES', 'SCHEMA_VERSION', 'main']

SCHEMA_VERSION: Final = 1
_EXIT_REFUSED: Final = 2

#: How much of the finding's own file the first turn carries: the judge reads it at the HEAD, never only the range's
#: increment (dev-lab #7 round 3); past this it is cut, and the judge asks for the rest.
HELD_BYTES: Final = 40_000

#: The verdicts a judge may reach; anything else is not judged, and stays ``present``.
OUTCOMES: Final = ('fixed', 'present', 'false-positive')

_KEYS: Final = ('id', 'severity', 'security', 'rule', 'file', 'anchor', 'summary', 'detail', 'suggestion', 'claim')

_INSTRUCTIONS: Final = 'Judge the one finding the user turn describes; respond with the object the schema declares.'

_SCHEMA: Final = {
    'type': 'object',
    'required': ['outcome', 'evidence'],
    'properties': {
        'outcome': {'type': 'string', 'enum': list(OUTCOMES)},
        'evidence': {'type': 'string', 'description': 'one or two sentences citing file:line or a test'},
    },
}

_PROMPT: Final = """You are an independent reviewer RE-JUDGING one open review finding against the head of {range_}.
Explore the head before deciding: each turn but the last you may answer with `requests` (diff, show, grep, list of
paths or patterns) and they are answered next turn; you have {turns} turns.

Decide ONE outcome:
- fixed: the defect no longer exists at the head. Cite where (file:line) it is gone.
- false-positive: it never was a defect. Cite the code or test that proves it; the author's claim alone is not proof.
- present: the defect is still there, or you cannot establish either of the above.
Answer with `outcome` and `evidence` (one or two sentences citing file:line or a test).

The finding:
- rule: {rule}
- file: {file}
- anchor: {anchor}
- summary: {summary}
- detail: {detail}

The author's claim that this is not a defect: {claim}

Principles:
{principles}

Registry rule ids: {rules}

The file {file} as the head holds it (read whatever the range changed; cut when long):
{held}

The change map:
{map}
"""


def _findings(path: str) -> list[dict]:
    """The ``--findings`` file's findings; ``ValueError``/``TypeError`` when it is not that shape."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as bad:
        msg = f'{path} is not JSON: {bad}'
        raise ValueError(msg) from None
    rows = data.get('findings') if isinstance(data, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) and set(_KEYS) <= set(row) for row in rows):
        msg = f'{path} is not {{"findings": [{{{", ".join(_KEYS)}}}]}}'
        raise TypeError(msg)
    return rows


def _prompt(
    finding: Mapping, *, range_: str, mapped: str, held: str, principles: Sequence[Mapping], rules: Sequence[str]
) -> str:
    listed = '\n'.join(f'- {p["id"]} ({p["name"]}): {p["statement"]}' for p in principles) or '(none)'
    return _PROMPT.format(
        range_=range_, turns=TURNS, rule=finding['rule'], file=finding['file'], anchor=finding['anchor'],
        summary=finding['summary'], detail=finding['detail'], claim=finding['claim'] or '(none)',
        principles=listed, rules=', '.join(rules) or '(none)', map=mapped, held=held,
    )  # fmt: skip


def _verdict(fid: str, reply: Mapping) -> dict:
    """The verdict *reply* reaches; a failure or an invalid outcome is ``present``, ``not judged`` (fails closed)."""
    if not reply.get('ok'):
        return {'id': fid, 'outcome': 'present', 'evidence': f'not judged: {reply.get("kind") or "call failed"}'}
    result = reply.get('result')
    outcome = result.get('outcome') if isinstance(result, dict) else None
    if outcome not in OUTCOMES:
        return {'id': fid, 'outcome': 'present', 'evidence': f'not judged: no valid outcome ({outcome!r})'}
    return {'id': fid, 'outcome': outcome, 'evidence': str(result.get('evidence') or '')}


def main(argv: Sequence[str] | None = None, *, call: Callable[[Mapping], Mapping] = answer) -> int:
    """Judge every finding against the range's head and print the verdicts JSON; exit 2 on a refused invocation."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.review.judge', description=__doc__.split('\n', 1)[0])
    parser.add_argument('--range', required=True, dest='range_', help='the git range whose head is judged')
    parser.add_argument('--repo', default='.', help='the repository the range is in (default: the working directory)')
    parser.add_argument('--model', required=True, help='the model id for the provider the method names none for')
    parser.add_argument('--rules', help='the registry rule ids, comma-separated (required)')
    parser.add_argument('--principles', help='a JSON file {"principles": [{id, name, statement}]} (required)')
    parser.add_argument('--findings', required=True, help='a JSON file {"findings": [...]}: the open findings')
    args = parser.parse_args(argv)
    try:
        method = load_method(METHOD_DIR, principles=principle_list(args.principles), registry=rule_ids(args.rules))
        findings = _findings(args.findings)
        rows = numstat(args.range_, args.repo)
    except (FileNotFoundError, UnknownMethodVersion, ValueError, TypeError) as refused:
        sys.stderr.write(f'refused: {refused}\n')
        return _EXIT_REFUSED

    mode = method.spec['run_modes']['headless']
    provider = (configured_providers({}, 'headless') or mode['providers'])[0]
    request = {'verb': 'respond', 'provider': provider, 'model': mode['models'].get(provider, args.model),
               'task': REVIEW_TASK, 'instructions': _INSTRUCTIONS, 'schema': _SCHEMA}  # fmt: skip
    mapped = change_map(args.range_, rows)

    def explore(asked: Mapping) -> str:
        return serve(asked, range_=args.range_, repo=args.repo, rows=rows)

    verdicts = []
    for finding in findings:
        held = serve({'show': [finding['file']]}, range_=args.range_, repo=args.repo, rows=rows, budget=HELD_BYTES)
        prompt = _prompt(finding, range_=args.range_, mapped=mapped, held=held, principles=method.principles,
                         rules=method.registry)  # fmt: skip
        verdicts.append(_verdict(str(finding['id']), explored(call, request, prompt, explore, TURNS)))
    sys.stdout.write(json.dumps({'schema_version': SCHEMA_VERSION, 'verdicts': verdicts}, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
