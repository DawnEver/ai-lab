"""The headless run mode: plan the reviewers, render each prompt, ``respond`` through ai-lab, merge.

``method.json``'s ``run_modes.headless`` is the configuration: its providers (overridable), the contexts
and call modes it can execute, its per-provider models. A reviewer is ``skipped`` when its context has
nothing to review, ``unavailable`` when its provider is not in the available set (never substituted),
``unsupported`` when the run mode cannot make its call; a failed call's ``kind`` is its status.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Collection, Mapping, Sequence

from ai_lab.review.method import Method
from ai_lab.review.rules import merge, pick, render_prompt

__all__ = ['REVIEW_TASK', 'run']

#: The key task every review call is made under (its own key slots: ``auth login <provider> --task``).
REVIEW_TASK = 'pr-review'

#: The system turn of every review call. The rendered prompt (the profile's brief plus the diff) is the
#: CONTEXT -- the user turn -- because a wire builds its user message from the context, and an empty one
#: is refused (Anthropic: ``user messages must have non-empty content``).
_INSTRUCTIONS = 'Review the change the user turn describes; respond with the findings object the schema declares.'


def _plan(method: Method, mode: Mapping, rng: random.Random) -> list[str]:
    spec = method.spec['pick']
    weights = [(p['key'], p['weight']) for p in method.profiles if p['context'] in mode['contexts']]
    chosen = pick(method, weights, spec['count'], [rng.random() for _ in range(spec['count'])])
    if spec['assignment'] == 'shuffled':
        rng.shuffle(chosen)
    return chosen


def _schema(method: Method) -> dict:
    item = {k: v for k, v in method.finding_schema.items() if k not in {'$schema', 'title'}}
    return {'type': 'object', 'required': ['findings'], 'properties': {'findings': {'type': 'array', 'items': item}}}


def _result(name: str, reply: Mapping) -> dict:
    if reply['ok']:
        findings = (reply['result'] or {}).get('findings')
        return {'name': name, 'findings': findings}
    return {'name': name, 'error_kind': reply['kind'], 'text': reply['error']}


def run(
    method: Method,
    *,
    range_: str,
    diff: str,
    model: str,
    providers: Sequence[str] | None,
    available: Collection[str],
    call: Callable[[Mapping], Mapping],
    rng: random.Random,
    date: str,
) -> dict:
    """The findings JSON of one headless review of *diff* (the change *range_* names).

    *call* is ``ai_lab.call.answer`` (or a stand-in): one request dict in, one classified reply out.
    *model* serves every provider the run mode's ``models`` does not name.
    """
    mode = method.spec['run_modes']['headless']
    providers = list(providers or mode['providers'])
    manifest = {'mode': 'review' if diff.strip() else 'empty', 'range': range_, 'excludedSummary': '', 'diff': diff}
    schema = _schema(method)
    keys = _plan(method, mode, rng)
    reviewers, results = [], []
    for i, key in enumerate(keys):
        provider = providers[i % len(providers)]
        rendered = render_prompt(method, method.profile(key), manifest)
        if 'status' in rendered:
            status = rendered['status']
        elif provider not in available:
            status = 'unavailable'
        elif rendered['call_mode'] not in mode['call_modes']:
            status = 'unsupported'
        else:
            status = 'pending'
        reviewers.append({'name': key, 'provider': provider, 'status': status})
        if status != 'pending':
            continue
        reply = call({
            'verb': 'respond', 'provider': provider, 'model': mode['models'].get(provider, model),
            'task': REVIEW_TASK, 'instructions': _INSTRUCTIONS,
            'context': [{'kind': 'text', 'text': rendered['prompt']}], 'schema': schema,
        })  # fmt: skip
        results.append(_result(key, reply))
    return merge(method, range_=range_, date=date, profiles=keys, reviewers=reviewers, results=results)
