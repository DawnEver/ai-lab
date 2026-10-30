"""A test calling a paid model needs a credential: it is the ``live`` need-set, never ``tests/base``.

A hosted runner has no credential, so the ``live`` part is posted absent there -- on record, not failed and not
skipped -- and a host declaring the resource produces it.
"""

import tomllib
from pathlib import Path

from dev_lab.dev.unproduced import unproduced
from lab_commons.hpc.needsets import BASE, read_vocabulary, selecting

from tests.live import test_live, test_openai

PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'


def test_live_is_a_declared_resource_so_base_excludes_it():
    vocabulary = read_vocabulary(PYPROJECT.read_text(encoding='utf-8'))
    assert 'live' in vocabulary.resources
    assert 'not live' in selecting(vocabulary, BASE)


def test_no_default_marker_expression_hides_live_from_part_derivation():
    pytest_options = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['tool']['pytest']['ini_options']
    assert '-m' not in pytest_options.get('addopts', '')


def test_the_openai_call_is_its_own_unproduced_resource_so_live_runs_without_its_key():
    """User decision 2026-10-30: no OpenAI key is a repo secret, so its test leaves ``live`` and runs nowhere."""
    text = PYPROJECT.read_text(encoding='utf-8')
    assert 'openai' in read_vocabulary(text).resources
    assert 'openai' in unproduced(text)
    marks = {test_openai.pytestmark.name}
    assert marks == {'openai'}, 'it is not in live, so live never falsely fails on a missing OpenAI key'
    assert not hasattr(test_live, 'test_openai_decisions_native')
    assert test_live.pytestmark.name == 'live', 'PLANTED CONTROL: the rest stays live'
