"""``python -m ai_lab.review.plan``: one review run planned from the method -- a request on stdin, the plan on stdout.

The planner a host executor (sharp-review's Claude/Codex fan-out) runs before it fans the reviewers out: which
profiles review, which provider runs each, the exact prompt each receives (PLAN-a-short-robust-toolchain P4: the
host keeps no method logic). The request is ``{manifest, run_id, started_at, date, seed}`` plus the optional
``config`` (the project's ``.claude/sharp-review.json``), ``sources``, ``forced``, ``run_mode`` (default
``interactive``), ``requested``, ``available`` and ``rands`` (``{pick, shuffle}`` draws; drawn here when absent).
The manifest is ``{mode, range, excludedSummary, diff, manifest, stats}``. A refused request -- not JSON, not an
object, lacking a required key, naming an unknown run mode or profile -- exits 2 with ``refused: <why>`` on stderr.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Collection, Mapping, Sequence
from typing import Final

from ai_lab.review.method import METHOD_DIR, Method, UnknownMethodVersion, load_method, principle_list, rule_ids
from ai_lab.review.rules import pick, render_prompt

__all__ = ['build_plan', 'effective_profiles', 'eligible_weights']

_EXIT_REFUSED: Final = 2
_REQUIRED: Final = ('manifest', 'run_id', 'started_at', 'date', 'seed')
_CUSTOM_REQUIRED: Final = ('key', 'source', 'context', 'scope')
#: What a custom profile is unless it says otherwise.
_CUSTOM_DEFAULTS: Final = {'framing': None, 'weight': 0.1}


def effective_profiles(method: Method, config: Mapping) -> dict[str, dict]:
    """The method's profiles overlaid by the project config: ``customProfiles`` add or replace, weights retune.

    An invalid custom profile is refused by name, never skipped.
    """
    profiles = {row['key']: dict(row) for row in method.profiles}
    for row in config.get('customProfiles') or []:
        missing = [k for k in _CUSTOM_REQUIRED if row.get(k) is None]
        if missing:
            msg = f'customProfiles entry {row.get("key", row)!r} lacks {", ".join(missing)}'
            raise ValueError(msg)
        if row['context'] not in method.spec['contexts']:
            msg = f'customProfiles entry {row["key"]!r} names unknown context {row["context"]!r}'
            raise ValueError(msg)
        profiles[row['key']] = {'label': row['key'], **_CUSTOM_DEFAULTS, **row}
    for key, weight in (config.get('profileWeights') or {}).items():
        if key in profiles:
            profiles[key] = {**profiles[key], 'weight': weight}
    return profiles


def eligible_weights(
    profiles: Mapping[str, Mapping], *, sources: Collection[str] = (), contexts: Collection[str] | None = None
) -> list[tuple[str, float]]:
    """``(key, weight)`` of the profiles eligible this round: source fired (all when none) and a permitted context."""
    return [
        (p['key'], p['weight'])
        for p in profiles.values()
        if (not sources or p['source'] in sources) and (contexts is None or p['context'] in contexts)
    ]


def _shuffle(keys: list[str], rands: Sequence[float]) -> list[str]:
    out = list(keys)
    for i in range(len(out) - 1, 0, -1):
        j = int(rands[i] * (i + 1))
        out[i], out[j] = out[j], out[i]
    return out


def _providers(mode: Mapping, name: str, *, requested: Sequence[str], config: Mapping, available: Collection[str],
               seed: int) -> list[str]:  # fmt: skip
    """Most specific first: *requested*, the config's ``providers.<run mode>``, the run mode's list, else its roster."""
    configured = (config.get('providers') or {}).get(name)
    if requested:
        return list(requested)
    if configured:
        return list(configured)
    spec = mode['providers']
    if isinstance(spec, list):
        return list(spec)
    roster = sorted(available)
    if not roster:
        return []
    start = seed % len(roster)
    return (roster[start:] + roster[:start])[: spec['count']]


def report_path(date: str) -> str:
    """Where the ``report`` sink writes run *date*'s (``YYYY-MM-DD``) review, relative to the reviewed checkout.

    Regenerable program output, so it lives under ``output/logs/<yy>/<mm>/<dd>/sharp-review/`` and never in a
    tracked memory path; a finding a person keeps is moved into memory by that person.
    """
    yyyy, mm, dd = date.split('-')
    return f'output/logs/{yyyy[2:]}/{mm}/{dd}/sharp-review/sharp-review.md'


def build_plan(
    method: Method,
    *,
    manifest: Mapping,
    rands: Mapping[str, Sequence[float]],
    run_id: str,
    started_at: str,
    date: str,
    seed: int,
    config: Mapping | None = None,
    sources: Sequence[str] = (),
    forced: str | None = None,
    run_mode: str = 'interactive',
    requested: Sequence[str] = (),
    available: Collection[str] = (),
) -> dict:
    """The plan of one run: its profiles, and per reviewer a provider, a status and (pending) the prompt."""
    config = config or {}
    modes = method.spec['run_modes']
    if run_mode not in modes:
        msg = f'unknown run mode {run_mode}; known: {", ".join(modes)}'
        raise ValueError(msg)
    mode = modes[run_mode]
    profiles = effective_profiles(method, config)
    if forced:
        if forced not in profiles:
            msg = f'unknown profile {forced}; known: {", ".join(profiles)}'
            raise ValueError(msg)
        keys = [forced]
    else:
        weights = eligible_weights(profiles, sources=sources, contexts=mode.get('contexts'))
        keys = pick(method, weights, method.spec['pick']['count'], rands['pick'])
    if method.spec['pick']['assignment'] == 'shuffled':
        keys = _shuffle(keys, rands['shuffle'])
    providers = _providers(mode, run_mode, requested=requested, config=config, available=available, seed=seed)
    reviewers = []
    for i, key in enumerate(keys):
        provider = providers[i % len(providers)] if providers else None
        row = {'name': key, 'provider': provider, 'profile': key}
        if provider is None or provider not in available:
            reviewers.append({**row, 'status': 'unavailable'})
            continue
        rendered = render_prompt(method, profiles[key], manifest)
        if 'status' in rendered:
            reviewers.append({**row, 'status': rendered['status']})
        elif rendered['call_mode'] not in mode['call_modes']:
            reviewers.append({**row, 'status': 'unsupported', 'call_mode': rendered['call_mode']})
        else:
            model = {'model': mode['models'][provider]} if provider in mode['models'] else {}
            reviewers.append({**row, 'status': 'pending', 'call_mode': rendered['call_mode'], **model,
                              'prompt': rendered['prompt']})  # fmt: skip
    return {
        'schema_version': method.schema_version,
        'run_id': run_id,
        'started_at': started_at,
        'date': date,
        'run_mode': run_mode,
        'range': manifest.get('range'),
        'seed': seed,
        'stats': manifest.get('stats'),
        'excludedSummary': manifest.get('excludedSummary'),
        'synthesis': mode['synthesis'],
        'sink': mode['sink'],
        'report': report_path(date),
        'high_confidence': method.spec['merge']['confidence'][0]['label'],
        'profiles': [{'key': k, 'label': profiles[k].get('label')} for k in keys],
        'reviewers': reviewers,
    }


def _request(text: str, count: int) -> dict:
    """The keywords of :func:`build_plan` *text* holds; ``ValueError``/``TypeError`` naming what is wrong."""
    try:
        request = json.loads(text)
    except json.JSONDecodeError as bad:
        msg = f'the plan request on stdin is not JSON: {bad}'
        raise ValueError(msg) from None
    if not isinstance(request, dict):
        msg = f'the plan request on stdin is a {type(request).__name__}, not an object'
        raise TypeError(msg)
    missing = [k for k in _REQUIRED if k not in request]
    if missing:
        msg = f'the plan request on stdin lacks {missing}'
        raise ValueError(msg)
    rng = random.SystemRandom()
    draws = request.pop('rands', None) or {k: [rng.random() for _ in range(count)] for k in ('pick', 'shuffle')}
    return {**request, 'rands': draws}


def main(argv: Sequence[str] | None = None) -> int:
    """Plan the run the request on stdin describes and print the plan; exit 2 when refused."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.review.plan', description=__doc__.split('\n', 1)[0])
    parser.add_argument(
        '--method', default=METHOD_DIR, help='a method directory read in place (default: the packaged one)'
    )
    parser.add_argument(
        '--rules', help='the registry rule ids, comma-separated (required: the caller owns the registry)'
    )
    parser.add_argument(
        '--principles', help='a JSON file {"principles": [{id, name, statement}]} (required: the caller owns them)'
    )
    args = parser.parse_args(argv)
    try:
        method = load_method(args.method, principles=principle_list(args.principles), registry=rule_ids(args.rules))
        plan = build_plan(method, **_request(sys.stdin.read(), method.spec['pick']['count']))
    except (FileNotFoundError, UnknownMethodVersion, ValueError, TypeError) as refused:
        sys.stderr.write(f'refused: {refused}\n')
        return _EXIT_REFUSED
    sys.stdout.write(json.dumps(plan, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
