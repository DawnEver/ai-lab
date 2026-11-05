"""A test calling a paid model needs a credential: it is the ``live`` need-set, never ``tests/base``.

User ruling 2026-11-05: every test completes on GitHub Actions alone. ``live`` -- the Anthropic and the OpenAI calls --
runs on its own hosted row, whose setup step imports each key from a repository secret through
``ai_lab.auth login --from-file -`` (the declaration is ``[tool.dev_lab.parts] hosted``). Nothing is unproduced.
"""

import tomllib
from pathlib import Path

from dev_lab.dev.jobparts import hosted_parts
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


def test_every_resource_is_produced_on_actions():
    """User ruling 2026-11-05: no test waits on, or is skipped for want of, a workstation or a key."""
    text = PYPROJECT.read_text(encoding='utf-8')
    assert unproduced(text) == frozenset()
    assert set(read_vocabulary(text).resources) == {'live'}


def test_live_holds_both_providers_keys_from_its_own_repo_secrets():
    """Never the org ``ANTHROPIC_API_KEY`` (being revoked): the live row names its own repo secrets."""
    live = hosted_parts(PYPROJECT.read_text(encoding='utf-8'))['live']
    assert set(live.secrets) == {'AI_LAB_LIVE_ANTHROPIC_KEY', 'AI_LAB_LIVE_OPENAI_KEY'}
    for key_name in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY'):
        assert f'/{key_name}=/' in live.setup, f'the setup renames a secret to {key_name}'
    assert test_openai.pytestmark.name == test_live.pytestmark.name == 'live'
