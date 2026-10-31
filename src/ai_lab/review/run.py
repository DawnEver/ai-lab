"""The headless run mode: plan the reviewers, render each prompt for the change's tier, let each explore, merge.

A reviewer is given its tier's brief (:func:`ai_lab.review.explore.brief`: the whole diff when small, the map when
medium, the directory map and the commit messages when huge); each turn it answers with ``findings`` or with
``requests`` the caller's *explore* answers, for at most ``turns`` turns, the last of which must answer.

``method.json``'s ``run_modes.headless`` is the capability: its providers (the shipped default config's, else
overridable), the contexts
and call modes it can execute, its per-provider models. A reviewer is ``skipped`` when its context has
nothing to review, ``unavailable`` when its provider is not in the available set (never substituted),
``unsupported`` when the run mode cannot make its call; a failed call's ``kind`` is its status and its
``error`` text (which never carries a key) is the reviewer's ``error``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Final

from ai_lab.review.explore import REQUESTS, TURNS
from ai_lab.review.method import DEFAULT_CONFIG, Method, configured_providers
from ai_lab.review.rules import merge, pick, render_prompt

__all__ = ['LENSES', 'REVIEW_TASK', 'run']

#: The key task every review call is made under (its own key slots: ``auth login <provider> --task``).
REVIEW_TASK = 'pr-review'

#: The system turn of every review call. The rendered prompt (the profile's brief plus the change map) is the
#: CONTEXT -- the user turn -- because a wire builds its user message from the context, and an empty one
#: is refused (Anthropic: ``user messages must have non-empty content``).
_INSTRUCTIONS = 'Review the change the user turn describes; respond with the findings object the schema declares.'


#: How many reviewers each lens plans, from the method's own count and the eligible profiles: ``strict``
#: every eligible profile, ``standard`` the method's count, ``light`` one.
LENSES: Final[Mapping[str, Callable[[int, int], int]]] = {
    'strict': lambda _count, eligible: eligible,
    'standard': lambda count, _eligible: count,
    'light': lambda _count, _eligible: 1,
}


def _plan(method: Method, mode: Mapping, rng: random.Random, lens: str) -> list[str]:
    if lens not in LENSES:
        msg = f'unknown lens {lens!r}: one of {sorted(LENSES)}'
        raise ValueError(msg)
    spec = method.spec['pick']
    weights = [(p['key'], p['weight']) for p in method.profiles if p['context'] in mode['contexts']]
    count = LENSES[lens](spec['count'], sum(1 for _, weight in weights if weight > 0))
    chosen = pick(method, weights, count, [rng.random() for _ in range(count)])
    if spec['assignment'] == 'shuffled':
        rng.shuffle(chosen)
    return chosen


def _schema(method: Method) -> dict:
    item = {k: v for k, v in method.finding_schema.items() if k not in {'$schema', 'title'}}
    return {'type': 'object', 'required': ['findings'], 'properties': {'findings': {'type': 'array', 'items': item}}}


def _asking(schema: Mapping) -> dict:
    """The schema of a turn that may ask: ``findings`` (done) or ``requests`` (:data:`REQUESTS`), neither required."""
    asks = {
        kind: {'type': 'array', 'items': {'type': 'string'}, 'description': what} for kind, what in REQUESTS.items()
    }
    findings = schema['properties']['findings']
    return {'type': 'object', 'properties': {'findings': findings, 'requests': {'type': 'object', 'properties': asks}}}


def _explored(call: Callable[[Mapping], Mapping], request: Mapping, prompt: str, explore: Callable[[Mapping], str],
              turns: int) -> Mapping:  # fmt: skip
    """The reply that ends one reviewer's exploration: its findings, a failure, or the last turn's answer."""
    context = [{'kind': 'text', 'text': prompt}]
    for turn in range(1, turns + 1):
        last = turn == turns
        schema = request['schema'] if last else _asking(request['schema'])
        reply = call({**request, 'context': list(context), 'schema': schema})
        result = reply['result'] if reply['ok'] else None
        asked = result.get('requests') if isinstance(result, dict) else None
        if last or not asked:
            return reply
        left = turns - turn
        close = 'the next turn is the LAST: answer with findings' if left == 1 else f'{left} turn(s) left'
        said = f'Turn {turn}: you asked {json.dumps(asked, ensure_ascii=False)}\n{explore(asked)}\n({close}.)'
        context.append({'kind': 'text', 'text': said})
    return reply


def _result(name: str, reply: Mapping) -> dict:
    if reply['ok']:
        findings = (reply['result'] or {}).get('findings')
        return {'name': name, 'findings': findings}
    return {'name': name, 'error_kind': reply['kind'], 'error': reply['error']}


def run(
    method: Method,
    *,
    range_: str,
    brief: Mapping[str, str],
    explore: Callable[[Mapping], str],
    model: str,
    providers: Sequence[str] | None,
    available: Collection[str],
    call: Callable[[Mapping], Mapping],
    rng: random.Random,
    lens: str = 'standard',
    defaults: Mapping = DEFAULT_CONFIG,
    disposed: Mapping[str, str] | None = None,
    turns: int = TURNS,
) -> dict:
    """The findings JSON of one headless review of the change *range_* names, *brief* its tier's manifest fields.

    *explore* answers a turn's requests (:func:`ai_lab.review.explore.serve`); each reviewer has *turns* turns.

    *call* is ``ai_lab.call.answer`` (or a stand-in): one request dict in, one classified reply out.
    *model* serves every provider the run mode's ``models`` does not name. *lens* sets how many reviewers
    are planned (:data:`LENSES`); an unknown one is refused. An id in *disposed* is not re-raised (``merge``).
    """
    mode = method.spec['run_modes']['headless']
    providers = list(providers or configured_providers({}, 'headless', defaults) or mode['providers'])
    manifest = {**brief, 'range': range_, 'excludedSummary': '', 'turns': turns}
    schema = _schema(method)
    keys = _plan(method, mode, rng, lens)
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
        request = {
            'verb': 'respond', 'provider': provider, 'model': mode['models'].get(provider, model),
            'task': REVIEW_TASK, 'instructions': _INSTRUCTIONS, 'schema': schema,
        }  # fmt: skip
        results.append(_result(key, _explored(call, request, rendered['prompt'], explore, turns)))
    return merge(method, range_=range_, profiles=keys, reviewers=reviewers, results=results, disposed=disposed)
