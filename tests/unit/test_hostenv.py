"""ENV-IS-READ-ONCE: ai-lab reads the process environment in ai_lab.hostenv only, into a typed record."""

from __future__ import annotations

import os

import pytest

from ai_lab import hostenv


def test_the_record_reads_each_named_fact() -> None:
    env = {
        hostenv.HOME_ENV: '/h',
        hostenv.TAG_ENV: 'o/r#1',
        hostenv.SCOPE_ENV: 'campaign:T3',
        'USERNAME': 'alice',
        'USER': 'bob',
    }
    assert hostenv.context(env) == hostenv.Env(home='/h', spend_tag='o/r#1', scope='campaign:T3', user='alice')


def test_an_unsaid_fact_reads_empty_and_the_user_falls_back() -> None:
    assert hostenv.context({'USER': 'bob'}) == hostenv.Env(user='bob')
    assert hostenv.context({}).user == ''


def test_the_scope_is_set_and_cleared_for_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(hostenv.SCOPE_ENV, raising=False)
    hostenv.set_scope('campaign:T3')
    assert os.environ[hostenv.SCOPE_ENV] == 'campaign:T3'
    assert hostenv.context().scope == 'campaign:T3'
    hostenv.set_scope(None)
    assert hostenv.SCOPE_ENV not in os.environ


def test_the_child_environment_is_a_copy() -> None:
    child = hostenv.environment()
    child['AI_LAB_HOSTENV_PROBE'] = '1'
    assert 'AI_LAB_HOSTENV_PROBE' not in os.environ
