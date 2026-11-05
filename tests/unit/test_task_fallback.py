"""A task's call falls over to the next PROVIDER once its Anthropic slots are used up (user ruling 2026-11-05).

The chain is declared once, ``fallback = [...]`` on the provider row. Only an ACCOUNT-level failure of every slot
moves the call on; a per-call error never does, and a call without a task (the ``live`` tests) never does. Every
secret here is a fake.
"""

import json
import logging
from pathlib import Path

import pytest

from ai_lab import ProviderError, ResponseRequest, Text, catalogue, connect
from ai_lab.budget import ledger_file
from ai_lab.client import fallback_model
from tests.unit.test_one_account import _post

ANTHROPIC = 'api.anthropic.com'
DEEPSEEK = 'api.deepseek.com'
RAW = {'content': [{'type': 'text', 'text': '{"a": 1}'}], 'usage': {'input_tokens': 3, 'output_tokens': 2}}
REQUEST = ResponseRequest('i', (Text('t'),), {'type': 'object'})


def _keys(put):
    put(ANTHROPIC, 'sk-ant-r1', 'api-key-pr-review')
    put(ANTHROPIC, 'sk-ant-r2', 'api-key-pr-review-2')
    put(DEEPSEEK, 'sk-ds-r1', 'api-key-pr-review')


def _exhausted():
    return ProviderError('HTTP 402: credit balance too low', kind='exhausted')


def test_the_anthropic_row_declares_deepseek_as_its_fallback():
    assert catalogue()['anthropic'].fallback == ('deepseek',)


def test_exhausted_anthropic_falls_to_deepseek_and_the_ledger_says_so(credential_store, caplog):
    _keys(credential_store)
    post = _post(_exhausted(), _exhausted(), RAW)
    with caplog.at_level(logging.WARNING, logger='ai_lab.client'):
        got = connect('anthropic:claude-opus-5-5', task='pr-review', post=post).respond(REQUEST)
    assert got.value == {'a': 1}
    assert (got.provider, got.model, got.key_slot) == ('deepseek', 'deepseek-v4-flash', 1)
    assert post.calls[2].url == 'https://api.deepseek.com/anthropic/v1/messages'
    assert 'sk-ds-r1' in str(dict(post.calls[2].headers))
    assert post.calls[2].body['model'] == 'deepseek-v4-flash'
    row = json.loads(ledger_file().read_text(encoding='utf-8').splitlines()[-1])
    assert (row['provider'], row['fallback_from']) == ('deepseek', 'anthropic')
    logged = caplog.text
    assert 'anthropic' in logged and 'deepseek' in logged and 'pr-review' in logged and 'exhausted' in logged
    assert 'sk-' not in logged


def test_a_per_call_error_never_falls_over(credential_store):
    """PLANTED control: the same chain, a bad request -- the first slot's error is the answer."""
    _keys(credential_store)
    post = _post(ProviderError('HTTP 400: bad request', kind=''), RAW)
    with pytest.raises(ProviderError, match='bad request'):
        connect('anthropic:claude-opus-5-5', task='pr-review', post=post).respond(REQUEST)
    assert len(post.calls) == 1


def test_a_call_without_a_task_never_falls_over(credential_store):
    """The ``live`` tests connect without a task: a fallback there would hide the provider failing."""
    credential_store(ANTHROPIC, 'sk-ant-d1')
    credential_store(DEEPSEEK, 'sk-ds-d1')
    post = _post(_exhausted(), RAW)
    with pytest.raises(ProviderError, match='every key slot'):
        connect('anthropic:claude-opus-5-5', post=post).respond(REQUEST)
    assert len(post.calls) == 1


def test_a_fallback_without_the_tasks_key_reraises_the_anthropic_failure_naming_the_login(credential_store):
    credential_store(ANTHROPIC, 'sk-ant-r1', 'api-key-pr-review')
    post = _post(_exhausted())
    with pytest.raises(ProviderError, match='login deepseek --task pr-review') as raised:
        connect('anthropic:claude-opus-5-5', task='pr-review', post=post).respond(REQUEST)
    assert raised.value.kind == 'exhausted'


@pytest.mark.parametrize(('model', 'expected'), [
    ('claude-opus-5-5', 'deepseek-v4-flash'),
    ('claude-fable-5-1', 'deepseek-v4-pro'),
    ('claude-haiku-5-5', 'deepseek-v4-flash'),
    ('claude-unlisted', 'deepseek-v4-flash'),
])  # fmt: skip
def test_the_model_maps_by_role_through_the_rows_models(model, expected):
    table = catalogue()
    assert fallback_model(table['anthropic'], model, table['deepseek']) == expected


def test_the_live_tests_never_name_a_task():
    live = Path(__file__).resolve().parents[1] / 'live'
    assert not [p.name for p in live.glob('*.py') if 'task=' in p.read_text(encoding='utf-8')]
