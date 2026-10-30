"""A test calling a paid model needs a credential: it is the ``live`` need-set, never ``tests/base``.

A hosted runner has no credential, so the ``live`` part is posted absent there -- on record, not failed and not
skipped -- and a host declaring the resource produces it.
"""

import tomllib
from pathlib import Path

from lab_commons.hpc.needsets import BASE, read_vocabulary, selecting

PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'


def test_live_is_a_declared_resource_so_base_excludes_it():
    vocabulary = read_vocabulary(PYPROJECT.read_text(encoding='utf-8'))
    assert 'live' in vocabulary.resources
    assert 'not live' in selecting(vocabulary, BASE)


def test_no_default_marker_expression_hides_live_from_part_derivation():
    pytest_options = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['tool']['pytest']['ini_options']
    assert '-m' not in pytest_options.get('addopts', '')
