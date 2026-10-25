"""``ai_lab.review.run``: the headless run mode plans, renders, calls through ``respond`` and merges -- network-free.

Every document is checked against the method's own ``findings-output.schema.json`` by a validator of the
keywords that schema uses; an unknown keyword fails, so a schema change cannot pass unread.
"""

import re

import pytest

from ai_lab.review.method import load_method
from ai_lab.review.rules import merge
from ai_lab.review.run import run

_KEYWORDS = {'$schema', 'title', 'description', 'type', 'const', 'enum', 'required', 'properties',
             'additionalProperties', 'items', 'pattern', 'minLength', 'allOf', 'if', 'then', 'not'}  # fmt: skip
_TYPES = {'object': dict, 'array': list, 'string': str, 'integer': int, 'null': type(None)}
FINDING = {'severity': 'HIGH', 'file': 'a.py', 'summary': 'Null crashes parse', 'category': 'Bug'}


def _scalar_error(schema, value, at):
    if 'type' in schema:
        types = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        if not any(isinstance(value, _TYPES[t]) and not (t == 'integer' and isinstance(value, bool)) for t in types):
            return f'{at}: {value!r} is not {types}'
    if 'const' in schema and value != schema['const']:
        return f'{at}: {value!r} != {schema["const"]!r}'
    if 'enum' in schema and value not in schema['enum']:
        return f'{at}: {value!r} not in {schema["enum"]}'
    if 'pattern' in schema and not re.search(schema['pattern'], value):
        return f'{at}: {value!r} does not match {schema["pattern"]}'
    if 'minLength' in schema and len(value) < schema['minLength']:
        return f'{at}: {value!r} is shorter than {schema["minLength"]}'
    return None


def _errors(schema, value, at='$'):
    assert set(schema) <= _KEYWORDS, f'{at}: keywords {set(schema) - _KEYWORDS} unread by this validator'
    if (error := _scalar_error(schema, value, at)) is not None:
        return [error]
    errors = []
    if isinstance(value, dict):
        errors += [f'{at}: missing {k}' for k in schema.get('required', ()) if k not in value]
        props = schema.get('properties', {})
        if schema.get('additionalProperties') is False:
            errors += [f'{at}: unknown {k}' for k in set(value) - set(props)]
        errors += [e for k, sub in props.items() if k in value for e in _errors(sub, value[k], f'{at}.{k}')]
    if isinstance(value, list) and 'items' in schema:
        errors += [e for i, item in enumerate(value) for e in _errors(schema['items'], item, f'{at}[{i}]')]
    errors += [e for sub in schema.get('allOf', ()) for e in _errors(sub, value, at)]
    if 'if' in schema and not _errors(schema['if'], value, at):
        errors += _errors(schema.get('then', {}), value, at)
    if 'not' in schema and not _errors(schema['not'], value, at):
        errors.append(f'{at}: matches a forbidden shape')
    return errors


class Draws:
    """A deterministic stand-in for the run's random source: fixed draws, and a shuffle that reverses."""

    def __init__(self, *draws):
        self.draws = list(draws)

    def random(self):
        return self.draws.pop(0)

    def shuffle(self, items):
        items.reverse()


class FakeCall:
    """Stands in for ``ai_lab.call.answer``: records each request and returns the reply its provider maps to."""

    def __init__(self, replies):
        self.replies, self.requests = replies, []

    def __call__(self, request):
        self.requests.append(request)
        return self.replies[request['provider']]


def _ok(findings):
    return {'schema_version': 1, 'ok': True, 'kind': '', 'result': {'findings': findings}, 'usage': None,
            'key_slot': 1, 'error': ''}  # fmt: skip


def _failed(kind):
    return {'schema_version': 1, 'ok': False, 'kind': kind, 'result': None, 'usage': None, 'key_slot': 0,
            'error': 'it failed'}  # fmt: skip


@pytest.fixture(scope='module')
def method(method_dir):
    return load_method(method_dir)


def _run(method, call, **kw):
    kw.setdefault('providers', None)
    kw.setdefault('available', {'anthropic', 'openai'})
    return run(method, range_='main...HEAD', diff='+x = 1\n', model='claude-test', call=call,
               rng=Draws(0.7, 0.1), date='2026-10-20', **kw)  # fmt: skip


def test_the_headless_default_reviews_the_diff_through_anthropic_and_validates(method):
    call = FakeCall({'anthropic': _ok([FINDING, {'severity': 'NOPE'}])})
    doc = _run(method, call)
    assert _errors(method.output_schema, doc) == []
    assert len(doc['reviewers']) == method.spec['pick']['count']
    assert {r['provider'] for r in doc['reviewers']} == {'anthropic'}
    assert {r['status'] for r in doc['reviewers']} == {'ok'}
    assert all(r['dropped_findings'] == 1 for r in doc['reviewers'])
    assert doc['findings'][0]['confidence'].startswith('high-confidence')
    diff_keys = {p['key'] for p in method.profiles if p['context'] == 'diff'}
    assert set(doc['profiles']) <= diff_keys
    for request in call.requests:
        assert request['verb'] == 'respond'
        assert request['task'] == 'pr-review'
        assert request['model'] == 'claude-test'
        assert request['schema']['properties']['findings']['items']['required'] == ['severity', 'summary', 'category']
        # The rendered prompt is the user turn: an empty context is an empty user message, which the
        # Messages API refuses with a 400 (`user messages must have non-empty content`).
        assert [part['kind'] for part in request['context']] == ['text']
        assert '+x = 1' in request['context'][0]['text']


@pytest.mark.parametrize('kind', ['exhausted', 'auth', 'rate_limit', 'timeout', 'unsupported'])
def test_a_classified_failure_is_the_reviewers_status(method, kind):
    doc = _run(method, FakeCall({'anthropic': _failed(kind)}))
    assert _errors(method.output_schema, doc) == []
    assert {r['status'] for r in doc['reviewers']} == {kind}
    assert {r['error'] for r in doc['reviewers']} == {'it failed'}  # the failed call's error, verbatim
    assert doc['findings'] == []


def test_an_unclassified_failure_is_unavailable(method):
    doc = _run(method, FakeCall({'anthropic': _failed('')}))
    assert _errors(method.output_schema, doc) == []
    assert {(r['status'], r['error']) for r in doc['reviewers']} == {('unavailable', 'it failed')}


def test_an_unavailable_provider_is_never_substituted(method):
    call = FakeCall({'openai': _ok([FINDING])})
    doc = _run(method, call, providers=['ghost', 'openai'])
    assert _errors(method.output_schema, doc) == []
    assert [(r['provider'], r['status']) for r in doc['reviewers']] == [('ghost', 'unavailable'), ('openai', 'ok')]
    ghost, served = doc['reviewers']
    assert ghost['error'] == method.spec['reviewer_status']['reasons']['planned_final']['unavailable']
    assert 'error' not in served
    assert [r['provider'] for r in call.requests] == ['openai']


def test_an_empty_diff_is_skipped_without_a_call(method):
    call = FakeCall({})
    doc = run(method, range_='HEAD', diff='  \n', model='m', call=call, rng=Draws(0.5, 0.5), date='2026-10-20',
              providers=None, available={'anthropic'})  # fmt: skip
    assert _errors(method.output_schema, doc) == []
    assert {r['status'] for r in doc['reviewers']} == {'skipped'}
    assert not any('error' in r for r in doc['reviewers'])
    assert call.requests == []


def test_a_finding_lists_a_reviewer_once_by_name_whatever_its_provider(method):
    reviewers = [{'name': 'diff', 'provider': 'a', 'status': 'pending'},
                 {'name': 'diff', 'provider': 'b', 'status': 'pending'}]  # fmt: skip
    results = [{'name': 'diff', 'findings': [FINDING]}]
    doc = merge(method, range_='HEAD', date='2026-10-20', profiles=['diff'], reviewers=reviewers, results=results)
    assert doc['findings'][0]['reviewers'] == [{'name': 'diff', 'provider': 'a'}]
    assert doc['findings'][0]['confidence'] == method.spec['merge']['confidence'][-1]['label']
