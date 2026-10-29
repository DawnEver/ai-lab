"""The "what this change does" section: derived each run from the change's own record of intent, never stored.

The INTENT is what the caller's range already carries: its commits (sha, subject, body), the merge commits'
conflict-resolution lines (``<path>: <class> -- <why>``) and ``Merge-Audit:`` trailers, the ``.claude/memory/**``
documents the range adds or changes, and the PR body when given. Memory is read for intent only: it is not
reviewed, and reading it never widens the review's scope (the caller's paths decide that).

One model call writes the claims; every claim must cite a commit sha or memory path of the intent, and one
that cites nothing known is dropped. What tests were lost or altered and how conflicts were resolved are
copied from the merge commits verbatim, never from the model. A failed call keeps the summary: its claims
fall back to the commit subjects, each citing its sha, and ``error`` names the failure.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from typing import Final

__all__ = ['MEMORY_ROOT', 'change_summary', 'git_intent', 'parse_log']

#: Where intent documents live in every family repo.
MEMORY_ROOT: Final = '.claude/memory/'

_RECORD: Final = '\x1e'
_FIELD: Final = '\x00'
_AUDIT: Final = re.compile(r'^Merge-Audit:\s*(.+)$')
_RESOLUTION: Final = re.compile(r'^(\S+): (\S+) -- (.+)$')
_MEMORY_CHARS: Final = 4000

_INSTRUCTIONS: Final = (
    'Summarize what this change does and why, from the commits, merge resolutions, memory documents and PR '
    'body in the user turn. Every claim cites one or more commit shas or memory paths exactly as given; '
    'open_items are things the sources say remain to do. Do not review the code.'
)

_CITED: Final = {'type': 'array', 'items': {'type': 'string'}}
_SCHEMA: Final = {
    'type': 'object',
    'required': ['changes', 'open_items'],
    'properties': {
        'changes': {'type': 'array', 'items': {'type': 'object', 'required': ['claim', 'cites'],
                    'properties': {'claim': {'type': 'string'}, 'cites': _CITED}}},
        'open_items': {'type': 'array', 'items': {'type': 'object', 'required': ['item', 'cites'],
                       'properties': {'item': {'type': 'string'}, 'cites': _CITED}}},
    },
}  # fmt: skip


def parse_log(log: str) -> dict:
    """The commits, resolution lines and ``Merge-Audit`` lines of ``git log --format=%H%x00%B%x1e`` output."""
    commits, resolutions, audits = [], [], []
    for record in filter(str.strip, log.split(_RECORD)):
        sha, _, message = record.strip().partition(_FIELD)
        subject, _, body = message.strip().partition('\n')
        commits.append({'sha': sha, 'subject': subject.strip(), 'body': body.strip()})
        for line in body.splitlines():
            if audit := _AUDIT.match(line.strip()):
                audits.append({'sha': sha, 'line': audit.group(1).strip()})
            elif _RESOLUTION.match(line.strip()):
                resolutions.append({'sha': sha, 'line': line.strip()})
    return {'commits': commits, 'resolutions': resolutions, 'merge_audit': audits}


def _git(repo: str, *argv: str) -> str:
    git = shutil.which('git')
    if git is None:
        msg = 'git is not on PATH: the summary reads its intent with git'
        raise ValueError(msg)
    done = subprocess.run([git, '-C', repo, *argv], capture_output=True, text=True, encoding='utf-8', check=False)
    if done.returncode:
        msg = f'git {argv[0]} in {repo} failed: {done.stderr.strip()}'
        raise ValueError(msg)
    return done.stdout


def git_intent(range_: str, repo: str) -> dict:
    """The intent *range_* carries in *repo*: its log, and the memory documents it adds or changes (at its head)."""
    intent = parse_log(_git(repo, 'log', '--format=%H%x00%B%x1e', '--end-of-options', range_))
    head = re.split(r'\.\.\.?', range_)[-1] or 'HEAD'
    names = _git(repo, 'diff', '--name-only', '--diff-filter=AM', '--end-of-options', range_, '--', MEMORY_ROOT)
    intent['memory'] = [
        {'path': path, 'text': _git(repo, 'show', f'{head}:{path}')[:_MEMORY_CHARS]}
        for path in names.splitlines() if path.strip()
    ]  # fmt: skip
    return intent


def _brief(intent: Mapping) -> str:
    parts = ['COMMITS:'] + [f'{c["sha"]} {c["subject"]}\n{c["body"]}'.strip() for c in intent['commits']]
    parts += ['MERGE RESOLUTIONS:'] + [f'{r["sha"]} {r["line"]}' for r in intent['resolutions']]
    parts += ['MERGE-AUDIT:'] + [f'{a["sha"]} {a["line"]}' for a in intent['merge_audit']]
    parts += ['MEMORY (intent, not under review):'] + [f'--- {m["path"]}\n{m["text"]}' for m in intent['memory']]
    if intent.get('pr_body'):
        parts += ['PR BODY:', intent['pr_body']]
    return '\n'.join(parts)


def _cited(rows: object, text_key: str, known: set[str]) -> list[dict]:
    out = []
    for row in rows if isinstance(rows, list) else ():
        if not isinstance(row, Mapping) or not isinstance(row.get(text_key), str):
            continue
        cites = [c for c in row.get('cites') or () if c in known]
        if cites:
            out.append({text_key: row[text_key], 'cites': cites})
    return out


def change_summary(intent: Mapping, *, call: Callable[[Mapping], Mapping], provider: str, model: str) -> dict:
    """The summary section of *intent*, its claims written by one *call* (``ai_lab.call.answer`` or a fake)."""
    shas = [c['sha'] for c in intent['commits']]
    paths = [m['path'] for m in intent['memory']]
    known = set(shas) | set(paths)
    reply = call({
        'verb': 'respond', 'provider': provider, 'model': model, 'task': 'pr-review',
        'instructions': _INSTRUCTIONS, 'context': [{'kind': 'text', 'text': _brief(intent)}], 'schema': _SCHEMA,
    })  # fmt: skip
    result = (reply.get('result') or {}) if reply['ok'] else {}
    summary = {
        'changes': _cited(result.get('changes'), 'claim', known),
        'tests_altered': [{'line': a['line'], 'cites': [a['sha']]} for a in intent['merge_audit']],
        'resolutions': [{'line': r['line'], 'cites': [r['sha']]} for r in intent['resolutions']],
        'open_items': _cited(result.get('open_items'), 'item', known),
        'sources': {'commits': shas, 'memory': paths},
    }
    if not reply['ok']:
        summary['changes'] = [{'claim': c['subject'], 'cites': [c['sha']]} for c in intent['commits']]
        summary['error'] = f'{reply["kind"]}: {reply["error"]}'
    return summary
