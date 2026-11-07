"""A test calling a paid model needs a credential: it is the ``live`` need-set, never ``tests/base``.

User ruling 2026-11-05: every test completes on GitHub Actions alone. ``live`` -- the Anthropic and the OpenAI calls --
runs on its own hosted row, whose setup step imports the ``live`` task's key bundle from one repository secret
through ``ai_lab.auth login --task live --bundle -`` (declared as ``[tool.dev_lab.parts] hosted``).
Nothing is unproduced.
"""

import shlex
import tomllib
from pathlib import Path

from dev_lab.dev.jobparts import hosted_parts
from dev_lab.dev.taskkeys import loader
from dev_lab.dev.unproduced import unproduced
from lab_commons.hpc.needsets import BASE, read_vocabulary, selecting

from ai_lab.client import route
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


def test_live_holds_its_task_bundle_from_its_one_repo_secret():
    """Never the org ``ANTHROPIC_API_KEY``: the live row names ONE secret, the ``live`` task's key bundle.

    User ruling 2026-11-05: a task's secret is ``AI_LAB_KEYS_<TASK>`` and its setup is the bundle door; the
    ``live`` route's chain names every provider a live test calls, so the bundle may carry each one's keys.
    The row declares only its tasks; dev-lab renders each task's setup through ``taskkeys.loader``.
    """
    live = hosted_parts(PYPROJECT.read_text(encoding='utf-8'))['live']
    assert live.tasks == ('live',)
    assert set(live.secrets) == {'AI_LAB_KEYS_LIVE'}
    assert shlex.join(loader('python', 'live')) == 'python -I -m ai_lab.auth login --task live --bundle -'
    assert {name for name, _ in route('live').chain} >= {'anthropic', 'openai'}
    assert test_openai.pytestmark.name == test_live.pytestmark.name == 'live'
