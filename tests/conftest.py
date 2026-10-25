from pathlib import Path

import pytest
from lab_commons.box.machine import provided
from lab_commons.config import section
from lab_commons.credentials import store_credential

from ai_lab import Choice, DecisionRequest, Fields, Image, Level, Option, Predicate, Score, Text

PNG = b'\x89PNG\r\n\x1a\nfake'


@pytest.fixture
def questions():
    return (
        Predicate('worth_fea', 'Is this design worth an expensive FEA?'),
        Choice('fidelity', 'Which fidelity next?', (Option('drop', 'discard'), Option('fea2d'), Option('fea3d'))),
        Score('risk', 'Thermal risk', (Level('low'), Level('medium'), Level('high', 'magnet near demag'))),
    )


@pytest.fixture
def request_(questions):
    return DecisionRequest((Text('IPM rotor, 48 slots.'), Fields({'torque_nm': 327.0, 'ripple_pct': 4.8})), questions)


@pytest.fixture
def image_request(questions):
    return DecisionRequest((Text('flux plot'), Image(PNG)), questions)


@pytest.fixture(autouse=True)
def _private_home(tmp_path, monkeypatch):
    """Every test's ``AI_LAB_HOME`` (and so its token ledger) is its own; the box's real state is never touched."""
    monkeypatch.setenv('AI_LAB_HOME', str(tmp_path / 'home'))


@pytest.fixture
def credential_store(tmp_path, monkeypatch):
    """A throwaway git ``store`` helper and an empty user config root; the box's real store is never touched.

    Returns ``put(host, secret)``, which stores a FAKE secret where the client reads keys, with ``put.file``
    the store file a test can inspect.
    """
    store = tmp_path / 'git-credentials'
    config = tmp_path / 'gitconfig'
    config.write_text(f'[credential]\n\thelper = "store --file={store.as_posix()}"\n', encoding='utf-8')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(config))
    monkeypatch.setenv('GIT_CONFIG_SYSTEM', str(tmp_path / 'absent'))
    monkeypatch.setenv('AI_LAB_HOME', str(tmp_path / 'home'))

    def put(host, secret, username='api-key'):
        store_credential(host, username, secret)

    put.file = store
    return put


@pytest.fixture
def keys(credential_store):
    credential_store('api.openai.com', 'sk-test')
    credential_store('api.anthropic.com', 'ak-test')
    credential_store('api.typesafe.ai', 'k')
    return credential_store


#: The resource marker and machine-file capability that provide sharp-review's method data (a cc-market checkout).
METHOD_RESOURCE = 'sharp_review'


def method_path() -> Path | None:
    """The host's ``[capabilities.sharp_review] path`` (its cc-market ``sharp-review/method``), else ``None``."""
    declared = provided(section('capabilities')).get(METHOD_RESOURCE, {}).get('path')
    return Path(declared).expanduser() if declared else None


@pytest.fixture(scope='session')
def method_dir() -> Path:
    """sharp-review's method directory, read IN PLACE (never vendored); absent is a failure naming the remedy."""
    path = method_path()
    if path is None or not (path / 'method.json').is_file():
        pytest.fail(
            f'no sharp-review method at {path}: the machine file needs [capabilities.{METHOD_RESOURCE}] '
            'path = "<cc-market checkout>/sharp-review/method"'
        )
    return path
