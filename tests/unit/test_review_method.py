"""``ai_lab.review.method``: the method is read from the directory it is given, and an unknown version is refused."""

import json
import shutil

import pytest

from ai_lab.review.method import METHOD_SCHEMA_VERSIONS, Method, UnknownMethodVersion, load_method


@pytest.mark.sharp_review
def test_the_method_is_read_in_place(method_dir):
    method = load_method(method_dir)
    assert isinstance(method, Method)
    assert method.root == method_dir
    assert {p['key'] for p in method.profiles} >= {'diff', 'security', 'adversarial'}
    assert method.finding_schema['required'] == ['severity', 'summary', 'category']
    assert method.output_schema['properties']['schema_version'] == {'const': method.schema_version}
    assert method.text('prompts/diff.md').startswith('{{framing}}')


@pytest.mark.sharp_review
def test_an_unknown_schema_version_is_refused_by_name(method_dir, tmp_path):
    copy = tmp_path / 'method'
    shutil.copytree(method_dir, copy)
    data = json.loads((copy / 'method.json').read_text(encoding='utf-8'))
    data['schema_version'] = 99
    (copy / 'method.json').write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(UnknownMethodVersion, match=r'schema_version 99.*\[1\]'):
        load_method(copy)
    assert 99 not in METHOD_SCHEMA_VERSIONS


def test_a_directory_without_a_method_is_refused_naming_it(tmp_path):
    with pytest.raises(FileNotFoundError, match=r'method.json'):
        load_method(tmp_path)
