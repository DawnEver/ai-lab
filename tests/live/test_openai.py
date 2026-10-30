"""The OpenAI call: its own ``openai`` resource, declared unproduced -- no OpenAI key is a repository secret.

User decision 2026-10-30: it left ``live`` so the hosted live row never falsely fails on a missing key. It runs
nowhere until a producer holds that key; it is kept, never deleted.
"""

import pytest

from ai_lab import Basis, connect

pytestmark = pytest.mark.openai


def test_openai_decisions_native(request_):
    d = connect('openai:gpt-6-luna').decide(request_)
    assert d.basis is Basis.NATIVE
    assert d['fidelity'].probabilities is not None
