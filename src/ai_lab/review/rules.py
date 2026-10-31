"""The method's rules that are not data: pick, template rendering, reviewer status and merge.

Each rule reads its parameters from ``method.json``; the vocabulary it may name (an algorithm, a
precedence step) is fixed here, and an unknown one is refused rather than guessed. Behaviour is pinned by
the method's ``conformance/`` fixtures, shared with the Node executor.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Final

from ai_lab.review.method import Method

__all__ = ['finding_id', 'merge', 'pick', 'render', 'render_prompt', 'reviewer_status', 'valid_finding']

_PLACEHOLDER = re.compile(r'\{\{(\w+)\}\}')
_BLANK_RUN = re.compile(r'\n{3,}')
_CLAMP = 0.999999


def pick(method: Method, weights: Sequence[tuple[str, float]], count: int, rands: Sequence[float]) -> list[str]:
    """``count`` profile keys drawn from *weights* with the draws *rands*; the empty fallback when none is eligible."""
    spec = method.spec['pick']
    if spec['algorithm'] != 'weighted_without_replacement':
        msg = f'pick algorithm {spec["algorithm"]!r} is not one this executor implements'
        raise ValueError(msg)
    remaining = [(key, weight) for key, weight in weights if weight > 0]
    if not remaining:
        return [spec['empty_fallback']]
    chosen = []
    for rand in rands[: min(count, len(remaining))]:
        threshold = min(max(rand, 0.0), _CLAMP) * sum(w for _, w in remaining)
        index = len(remaining) - 1
        for i, (_, weight) in enumerate(remaining):
            threshold -= weight
            if threshold < 0:
                index = i
                break
        chosen.append(remaining.pop(index)[0])
    return chosen


def _value(value: object, join: str) -> str:
    if value is None:
        return ''
    if isinstance(value, list | tuple):
        return join.join(str(v) for v in value)
    return str(value)


def render(method: Method, template: str, variables: Mapping[str, object]) -> str:
    """*template* with each ``{{name}}`` substituted, blank runs collapsed and the result trimmed."""
    spec = method.spec['template']
    out = _PLACEHOLDER.sub(lambda m: _value(variables.get(m[1], spec['missing']), spec['list_join']), template)
    if spec['collapse_blank_lines']:
        out = _BLANK_RUN.sub('\n\n', out)
    return out.strip() if spec['trim'] else out


def _context_entry(method: Method, profile: Mapping, manifest: Mapping) -> dict:
    contexts = method.spec['contexts'][profile['context']]
    if 'any' in contexts:
        return contexts['any']
    mode = manifest['mode']
    if mode not in contexts:
        msg = f'context {profile["context"]!r} declares no entry for manifest mode {mode!r}'
        raise ValueError(msg)
    return contexts[mode]


def render_prompt(method: Method, profile: Mapping, manifest: Mapping) -> dict:
    """``{"call_mode", "prompt"}`` for *profile* on *manifest*, or the context's declared ``{"status"}``."""
    entry = _context_entry(method, profile, manifest)
    if 'template' not in entry:
        return {'status': entry['status']}
    schema = json.dumps(method.prompt_schema, separators=(',', ':'), ensure_ascii=False)
    variables = {
        'framing': profile.get('framing'),
        'scope': profile.get('scope'),
        'range': manifest.get('range'),
        'excluded_summary': manifest.get('excludedSummary'),
        'diff': manifest.get('diff'),
        'manifest': manifest.get('manifest'),
        'turns': manifest.get('turns'),
        'finding_schema': schema,
    }
    # method.json rules.list_item once per given principle, one per line; then the registry ids on one line.
    item = method.spec['rules']['list_item']
    variables['principle_list'] = '\n'.join(render(method, item, p) for p in method.principles)
    variables['registry_list'] = ', '.join(method.registry)
    for name, path in method.spec['template']['partials'].items():
        variables[name] = render(method, method.text(path), variables)
    return {'call_mode': entry['call_mode'], 'prompt': render(method, method.text(entry['template']), variables)}


def _placeholder(spec: Mapping, result: Mapping) -> bool:
    text = result.get('text')
    return isinstance(text, str) and any(re.search(p, text) for p in spec['placeholder_replies'])


#: Each precedence step the method may name: whether it applies to (planned, result), and the status it
#: decides when its step carries none (``error_kind`` decides the reported kind, if a known one).
_STEPS: Final = {
    'planned_final': lambda spec, planned, _: planned in spec['planned_final'],
    'missing_result': lambda _spec, _planned, result: result is None,
    'error_kind': lambda _spec, _planned, result: bool(result.get('error_kind')),
    'placeholder_reply': lambda spec, _planned, result: _placeholder(spec, result),
    'findings_array': lambda _spec, _planned, result: isinstance(result.get('findings'), list),
    'otherwise': lambda _spec, _planned, _result: True,
}


def reviewer_status(method: Method, planned: str, result: Mapping | None) -> str:
    """The reviewer's final status by ``reviewer_status.precedence``: the first step that applies decides."""
    return _decide(method, planned, result)[0]


def _decide(method: Method, planned: str, result: Mapping | None) -> tuple[str, str]:
    """The final status and the precedence step that decided it."""
    spec = method.spec['reviewer_status']
    for step in spec['precedence']:
        name, _, status = step.partition(':')
        if name not in _STEPS:
            msg = f'reviewer_status precedence step {name!r} is not one this executor implements'
            raise ValueError(msg)
        if not _STEPS[name](spec, planned, result):
            continue
        if name == 'planned_final':
            return planned, name
        if name == 'error_kind':
            kind = result['error_kind']
            return (kind if kind in spec['failure_kinds'] else 'unavailable'), name
        return status, name
    msg = 'reviewer_status.precedence ends without an otherwise step'
    raise ValueError(msg)


def _error(method: Method, status: str, step: str, reviewer: Mapping, result: Mapping | None) -> str | None:
    """Why a reviewer that is neither ok nor skipped has no findings (``reviewer_status.error``).

    The recorded error text -- the result's, else the planned reviewer's -- copied verbatim, else the
    deciding step's reason. The text is the call reply's ``error``, which names a failure without a key.
    """
    if status in {'ok', 'skipped'}:
        return None
    for recorded in ((result or {}).get('error'), reviewer.get('error')):
        if isinstance(recorded, str) and recorded.strip():
            return recorded
    reasons = method.spec['reviewer_status']['reasons']
    if step == 'planned_final':
        return reasons[step][status]
    return reasons[step].replace('{error_kind}', str((result or {}).get('error_kind', '')))


def _valid_value(rule: Mapping, value: object) -> bool:
    if rule.get('type') == 'string' and not (isinstance(value, str) and len(value) >= rule.get('minLength', 0)):
        return False
    if rule.get('type') == 'boolean' and not isinstance(value, bool):
        return False
    if rule.get('type') == 'array' and not (
        isinstance(value, list) and all(_valid_value(rule.get('items', {}), v) for v in value)
    ):
        return False
    return 'enum' not in rule or value in rule['enum']


def valid_finding(schema: Mapping, finding: object) -> bool:
    """*finding* against the finding schema's subset: an object, required non-null, types, ``minLength``, enums."""
    if schema.get('type') != 'object' or not isinstance(finding, dict):
        return False
    if any(finding.get(name) is None for name in schema.get('required', ())):
        return False
    return all(
        _valid_value(rule, finding[name]) for name, rule in schema.get('properties', {}).items() if name in finding
    )


def finding_id(spec: Mapping, finding: Mapping) -> str:
    """The finding's stable id (``merge.id``): a hash of its rule, file and anchor line, never a line number or date."""
    if spec['hash'] != 'sha256':
        msg = f'id hash {spec["hash"]!r} is not one this executor implements'
        raise ValueError(msg)
    text = '\n'.join(' '.join(str(finding.get(field, '')).split()) for field in spec['fields'])
    return f'{spec["prefix"]}{spec["separator"]}{hashlib.sha256(text.encode()).hexdigest()[: spec["length"]]}'


def merge(
    method: Method,
    *,
    range_: str,
    profiles: Sequence[str],
    reviewers: Sequence[Mapping],
    results: Sequence[Mapping],
    disposed: Mapping[str, str] | None = None,
) -> dict:
    """The findings JSON of one run: each planned reviewer's final status, and the findings grouped by stable id.

    A finding whose id is in *disposed* (id -> the reason it was disposed of earlier) is not re-raised; the
    document lists every disposed id with its reason under ``disposed`` instead.

    A group carries its first-seen fields, the most severe severity any reviewer gave and ``security`` when any
    said so (``merge.escalate``).
    """
    spec = method.spec['merge']
    order = spec['escalate']['severity_order']
    schema = method.finding_schema
    by_name = {r['name']: r for r in results}
    out_reviewers, groups = [], {}
    for reviewer in reviewers:
        result = by_name.get(reviewer['name'])
        status, step = _decide(method, reviewer['status'], result)
        dropped = 0
        if status == 'ok':
            who = {'name': reviewer['name'], 'provider': reviewer['provider']}
            for finding in result['findings']:
                if not valid_finding(schema, finding):
                    dropped += 1
                    continue
                key = finding_id(spec['id'], finding)
                group = groups.setdefault(key, {'finding': finding, 'severity': finding['severity'],
                                                'security': False, 'reviewers': []})  # fmt: skip
                group['severity'] = min(group['severity'], finding['severity'], key=order.index)
                group['security'] = group['security'] or finding['security']
                # A reviewer is its name: listed once per finding, whatever provider served it.
                if all(r['name'] != who['name'] for r in group['reviewers']):
                    group['reviewers'].append(who)
        out = {
            'name': reviewer['name'],
            'provider': reviewer['provider'],
            'status': status,
            'dropped_findings': dropped,
        }
        if (error := _error(method, status, step, reviewer, result)) is not None:
            out['error'] = error
        out_reviewers.append(out)
    disposed = disposed or {}
    doc = {
        'schema_version': method.schema_version,
        'range': range_,
        'profiles': list(profiles),
        'reviewers': out_reviewers,
        'findings': [_finding(spec, schema, key, group) for key, group in groups.items() if key not in disposed],
    }
    if disposed:
        doc['disposed'] = [{'id': key, 'reason': reason} for key, reason in disposed.items()]
    return doc


def _finding(spec: Mapping, schema: Mapping, key: str, group: Mapping) -> dict:
    count = len(group['reviewers'])
    label = next(c['label'] for c in spec['confidence'] if count >= c['min_reviewers'])
    first = group['finding']
    carried = {name: first.get(name, spec['defaults'].get(name, '')) for name in schema['properties']}
    return {
        'id': key,
        **carried,
        'severity': group['severity'],
        'security': group['security'],
        'status': spec['defaults']['status'],
        'confidence': label,
        'reviewers': group['reviewers'],
    }
