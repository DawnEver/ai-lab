"""``ai_lab.review.method``: the packaged method is the default, a directory given is read in place.

An unknown version is refused.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest
from dev_lab.dev.rules import RULES

import ai_lab.review
from ai_lab.review.method import METHOD_DIR, METHOD_SCHEMA_VERSIONS, Method, UnknownMethodVersion, load_method, main


def test_the_packaged_method_is_the_default():
    """The method is ai-lab's own data (PLAN-a-short-robust-toolchain P4): no argument loads the packaged copy."""
    assert Path(ai_lab.review.__file__).parent / 'method_data' == METHOD_DIR
    assert load_method().root == METHOD_DIR


def test_the_module_prints_the_packaged_method_directory(capsys):
    """``python -m ai_lab.review.method`` names the directory a host executor (sharp-review's planner) reads."""
    assert main([]) == 0
    assert capsys.readouterr().out == f'{METHOD_DIR}\n'


def test_the_method_is_read_in_place(method_dir):
    method = load_method(method_dir)
    assert isinstance(method, Method)
    assert method.root == method_dir
    assert {p['key'] for p in method.profiles} >= {'diff', 'security', 'adversarial'}
    assert method.finding_schema['required'] == [
        'severity',
        'security',
        'rule',
        'file',
        'anchor',
        'summary',
        'category',
    ]
    assert method.output_schema['properties']['schema_version'] == {'const': method.schema_version}
    assert method.text('prompts/diff.md').startswith('{{framing}}')


def test_an_unknown_schema_version_is_refused_by_name(method_dir, tmp_path):
    copy = tmp_path / 'method'
    shutil.copytree(method_dir, copy)
    data = json.loads((copy / 'method.json').read_text(encoding='utf-8'))
    data['schema_version'] = 99
    (copy / 'method.json').write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(UnknownMethodVersion, match=r'schema_version 99.*\[3\]'):
        load_method(copy)
    assert 99 not in METHOD_SCHEMA_VERSIONS


def test_the_rule_ids_are_the_principles_then_the_registry(method_dir, tmp_path):
    """The finding schema's ``rule`` enum is filled at load: ``principles.json`` ids, then the registry's.

    No file of the method lists them. PLANTED CONTROLS: a principle added to the data, and a rule injected as the
    registry, each reach the schema with no other edit; the default registry is the family's own.
    """
    method = load_method(method_dir, registry=('SOME-RULE',))
    field = method.spec['rules']['finding_field']
    raw = json.loads((method_dir / method.spec['files']['finding_schema']).read_text(encoding='utf-8'))
    assert 'enum' not in raw['properties'][field], 'a second copy of the ids'
    assert method.finding_schema['properties'][field]['enum'] == [*(p['id'] for p in method.principles), 'SOME-RULE']
    assert method.registry == ('SOME-RULE',)
    copy = tmp_path / 'method'
    shutil.copytree(method_dir, copy)
    data = json.loads((copy / 'principles.json').read_text(encoding='utf-8'))
    data['principles'].append({'id': 'fail-loud', 'name': 'Fail loud', 'statement': 'An unsupported case raises.'})
    (copy / 'principles.json').write_text(json.dumps(data), encoding='utf-8')
    assert 'fail-loud' in load_method(copy, registry=()).finding_schema['properties'][field]['enum']
    assert load_method(method_dir).registry == tuple(rule.id for rule in RULES)


def test_a_missing_registry_is_refused_by_name(method_dir, monkeypatch):
    """No registry, no method: a finding could not be validated, so the load refuses rather than guessing."""
    monkeypatch.setitem(sys.modules, 'dev_lab.dev.rules', None)
    with pytest.raises(ValueError, match=r'dev_lab\.dev\.rules'):
        load_method(method_dir)


def test_a_principle_id_that_is_also_a_registry_id_is_refused(method_dir):
    with pytest.raises(ValueError, match='occams-razor'):
        load_method(method_dir, registry=('occams-razor',))


def test_a_directory_without_a_method_is_refused_naming_it(tmp_path):
    with pytest.raises(FileNotFoundError, match=r'method.json'):
        load_method(tmp_path)


@pytest.mark.parametrize('choice', ['filled', 'unfilled', 'neither'])
def test_the_prompt_schema_is_the_one_method_json_names(method_dir, tmp_path, choice):
    """``rules.prompt_schema`` picks the prompt's schema; validation keeps the filled one. Unknown is refused."""
    copy = tmp_path / 'method'
    shutil.copytree(method_dir, copy)
    data = json.loads((copy / 'method.json').read_text(encoding='utf-8'))
    data['rules']['prompt_schema'] = choice
    (copy / 'method.json').write_text(json.dumps(data), encoding='utf-8')
    if choice == 'neither':
        with pytest.raises(ValueError, match=r"prompt_schema is 'neither'"):
            load_method(copy)
        return
    method = load_method(copy)
    field = data['rules']['finding_field']
    raw = json.loads((copy / data['files']['finding_schema']).read_text(encoding='utf-8'))
    assert 'enum' in method.finding_schema['properties'][field]
    assert method.prompt_schema == (method.finding_schema if choice == 'filled' else raw)
