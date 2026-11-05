"""The OpenAI call: part of ``live``, whose hosted row holds the OpenAI key from a repository secret.

User ruling 2026-11-05: every test completes on GitHub Actions alone, so the call that ran nowhere joined ``live``.
"""

import pytest

from ai_lab import Basis, connect

pytestmark = pytest.mark.live


def test_openai_decisions_native(request_):
    d = connect('openai:gpt-6-luna').decide(request_)
    assert d.basis is Basis.NATIVE
    assert d['fidelity'].probabilities is not None
