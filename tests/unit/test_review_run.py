"""``ai_lab.review.run``: the headless run mode plans, renders, calls through ``respond`` and merges -- network-free.

Every document is checked against the method's own ``findings-output.schema.json`` by a validator of the
keywords that schema uses; an unknown keyword fails, so a schema change cannot pass unread.
"""

import json
import re

import pytest

from ai_lab.review.method import load_method
from ai_lab.review.rules import merge
from ai_lab.review.run import run
from tests.review_principles import PRINCIPLES

_KEYWORDS = {'$schema', 'title', 'description', 'type', 'const', 'enum', 'required', 'properties',
             'additionalProperties', 'items', 'pattern', 'minLength', 'allOf', 'if', 'then', 'not'}  # fmt: skip
_TYPES = {'object': dict, 'array': list, 'string': str, 'integer': int, 'boolean': bool, 'null': type(None)}
FINDING = {'severity': 'HIGH', 'security': False, 'rule': 'occams-razor', 'file': 'a.py', 'anchor': 'return parse(x)',
           'summary': 'Null crashes parse', 'category': 'Bug'}  # fmt: skip


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


_MEDIUM = {'mode': 'medium', 'manifest': 'x.py  +1 -0', 'diff': '', 'commits': ''}
_EMPTY = {'mode': 'empty', 'manifest': '', 'diff': '', 'commits': ''}


def _nothing(_asked):
    return '(nothing)'


@pytest.fixture(scope='module')
def method(method_dir):
    return load_method(method_dir, principles=PRINCIPLES, registry=())


def _run(method, call, **kw):
    kw.setdefault('providers', None)
    kw.setdefault('available', {'anthropic', 'openai'})
    return run(method, range_='main...HEAD', brief=_MEDIUM, explore=_nothing, model='claude-test', call=call,
               rng=Draws(0.7, 0.1), **kw)  # fmt: skip


def test_the_headless_default_explores_the_change_through_anthropic_and_validates(method):
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
        assert request['schema']['properties']['findings']['items']['required'] == [
            'severity',
            'security',
            'rule',
            'file',
            'anchor',
            'summary',
            'category',
        ]
        # The rendered prompt is the user turn: an empty context is an empty user message, which the
        # Messages API refuses with a 400 (`user messages must have non-empty content`).
        assert [part['kind'] for part in request['context']] == ['text']
        assert 'x.py  +1 -0' in request['context'][0]['text'], 'the map, never a diff'


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


def test_an_empty_change_is_skipped_without_a_call(method):
    call = FakeCall({})
    doc = run(method, range_='HEAD', brief=_EMPTY,explore=_nothing, model='m', call=call, rng=Draws(0.5, 0.5),
              providers=None, available={'anthropic'})  # fmt: skip
    assert _errors(method.output_schema, doc) == []
    assert {r['status'] for r in doc['reviewers']} == {'skipped'}
    assert not any('error' in r for r in doc['reviewers'])
    assert call.requests == []


def test_a_finding_lists_a_reviewer_once_by_name_whatever_its_provider(method):
    reviewers = [{'name': 'diff', 'provider': 'a', 'status': 'pending'},
                 {'name': 'diff', 'provider': 'b', 'status': 'pending'}]  # fmt: skip
    results = [{'name': 'diff', 'findings': [FINDING]}]
    doc = merge(method, range_='HEAD', profiles=['diff'], reviewers=reviewers, results=results)
    assert doc['findings'][0]['reviewers'] == [{'name': 'diff', 'provider': 'a'}]
    assert doc['findings'][0]['confidence'] == method.spec['merge']['confidence'][-1]['label']


# -- The output schema is the versioned contract a findings-JSON reader relies on (moved from sharp-review's
# Node suite with the method, PLAN-a-short-robust-toolchain P4).


def test_every_merge_fixture_result_satisfies_the_output_schema(method):
    fixtures = sorted((method.root / method.spec['files']['conformance'] / 'merge').glob('*.json'))
    assert fixtures, 'a scan of no fixture passes anything'
    for path in fixtures:
        expected = json.loads(path.read_text(encoding='utf-8'))['expected']
        assert _errors(method.output_schema, expected) == [], path.name


def test_the_output_schema_enums_are_the_methods_own_not_a_second_copy(method):
    props = method.output_schema['properties']
    assert props['schema_version']['const'] == method.schema_version
    assert props['reviewers']['items']['properties']['status']['enum'] == method.spec['reviewer_status']['vocabulary']
    finding = props['findings']['items']['properties']
    assert finding['severity']['enum'] == method.finding_schema['properties']['severity']['enum']
    assert finding['category']['enum'] == method.finding_schema['properties']['category']['enum']
    assert finding['status']['const'] == method.spec['merge']['defaults']['status']


def test_the_statuses_that_must_carry_an_error_are_the_vocabulary_but_ok_and_skipped(method):
    _, must_say = method.output_schema['properties']['reviewers']['items']['allOf']
    vocabulary = method.spec['reviewer_status']['vocabulary']
    assert must_say['if']['properties']['status']['enum'] == [k for k in vocabulary if k not in {'ok', 'skipped'}]


def _eligible(method):
    contexts = method.spec['run_modes']['headless']['contexts']
    return [p['key'] for p in method.profiles if p['context'] in contexts and p['weight'] > 0]


def test_the_strict_lens_plans_every_eligible_profile(method):
    eligible = _eligible(method)
    doc = run(method, range_='a..b', brief=_MEDIUM, explore=_nothing, model='m', providers=None,
              available={'anthropic'}, call=FakeCall({'anthropic': _ok([])}), rng=Draws(*[0.0] * len(eligible)),
              lens='strict')  # fmt: skip
    assert sorted(doc['profiles']) == sorted(eligible)


def test_the_light_lens_plans_one_reviewer(method):
    doc = _run(method, FakeCall({'anthropic': _ok([])}), lens='light')
    assert len(doc['reviewers']) == 1


def test_the_standard_lens_is_the_methods_own_count(method):
    doc = _run(method, FakeCall({'anthropic': _ok([])}), lens='standard')
    assert len(doc['reviewers']) == method.spec['pick']['count']


def test_an_unknown_lens_is_refused(method):
    with pytest.raises(ValueError, match='lens'):
        _run(method, FakeCall({'anthropic': _ok([])}), lens='deep')
