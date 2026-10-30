from pathlib import Path

import pytest
from lab_commons.credentials import store_credential

from ai_lab import Choice, DecisionRequest, Fields, Image, Level, Option, Predicate, Score, Text, auth
from ai_lab.review.method import METHOD_DIR

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
def only_provider(monkeypatch):
    """``only(name)``: ``auth`` sees the one catalogue row a test's claim is about.

    Load, not behaviour: ``status`` asks git for every keyed host x (default + each task) x slot, one
    ``git credential`` process each (~60 over the full catalogue, ~7 s on Windows). A claim about one
    host's line is the same claim with the other rows absent.
    """

    def only(name):
        row = auth.catalogue()[name]
        monkeypatch.setattr(auth, 'catalogue', lambda: {name: row})

    return only


@pytest.fixture
def keys(credential_store):
    credential_store('api.openai.com', 'sk-test')
    credential_store('api.anthropic.com', 'ak-test')
    credential_store('api.typesafe.ai', 'k')
    return credential_store


@pytest.fixture(scope='session')
def method_dir() -> Path:
    """The packaged sharp-review method: ai-lab's own data since PLAN-a-short-robust-toolchain P4."""
    return METHOD_DIR
