"""Every wire's vendor failures land in ``ProviderError.kind``: exhausted, auth, rate limit, timeout.

Each row is a failure shape the vendor actually returns, driven through ``connect`` and the real
``transport.post`` (only ``urlopen`` is faked), so the classification a caller sees is the one tested.
Only the ACCOUNT kinds move a call to the next key slot; a timeout would repeat on any key.
"""

import io
import urllib.error

import pytest

from ai_lab import DecisionRequest, Predicate, ProviderError, ResponseRequest, Text, connect, transport
from ai_lab.errors import ACCOUNT_KINDS, ERROR_KINDS
from ai_lab.wire import WIRES

RESPOND = ResponseRequest('i', (Text('x'),), {'type': 'object'})
DECIDE = DecisionRequest((Text('x'),), (Predicate('p', 'q?'),))

#: ``spec -> wire`` for one provider of every shipped wire.
_SPECS = {
    'openai:gpt-x': 'openai_responses',
    'openai:gpt-6-luna': 'openai_decisions',
    'typesafe:t': 'typesafe',
    'anthropic:claude-x': 'anthropic_messages',
    'gemini:g': 'gemini',
    'openrouter:o': 'openai_chat',
    'deepseek:d': 'openai_chat_json',
}

#: ``(spec, status, body, kind)`` -- ``None`` status is a timeout before any status.
_FAILURES = [
    ('openai:gpt-x', 429, '{"error":{"type":"insufficient_quota"}}', 'exhausted'),
    ('openai:gpt-x', 401, '{"error":{"code":"invalid_api_key"}}', 'auth'),
    ('openai:gpt-x', 429, '{"error":{"type":"requests"}}', 'rate_limit'),
    ('openai:gpt-6-luna', 429, '{"error":{"type":"insufficient_quota"}}', 'exhausted'),
    ('openai:gpt-6-luna', 403, '{}', 'auth'),
    ('typesafe:t', 402, '{"error":"payment required"}', 'exhausted'),
    ('typesafe:t', 401, '{}', 'auth'),
    (
        'anthropic:claude-x',
        400,
        '{"error":{"message":"Your credit balance is too low to access the API"}}',
        'exhausted',
    ),
    ('anthropic:claude-x', 401, '{"error":{"type":"authentication_error"}}', 'auth'),
    ('anthropic:claude-x', 429, '{"error":{"type":"rate_limit_error"}}', 'rate_limit'),
    ('gemini:g', 400, '{"error":{"status":"INVALID_ARGUMENT","details":[{"reason":"API_KEY_INVALID"}]}}', 'auth'),
    ('gemini:g', 403, '{"error":{"status":"PERMISSION_DENIED"}}', 'auth'),
    ('gemini:g', 429, '{"error":{"status":"RESOURCE_EXHAUSTED"}}', 'rate_limit'),
    ('openrouter:o', 402, '{"error":{"message":"Insufficient credits"}}', 'exhausted'),
    ('openrouter:o', 401, '{}', 'auth'),
    ('deepseek:d', 402, '{"error":{"message":"Insufficient Balance"}}', 'exhausted'),
    ('deepseek:d', 401, '{"error":{"message":"Authentication Fails"}}', 'auth'),
    ('deepseek:d', 429, '{}', 'rate_limit'),
    *((spec, None, '', 'timeout') for spec in _SPECS),
]


@pytest.fixture
def every_key(keys):
    for host in ('generativelanguage.googleapis.com', 'openrouter.ai', 'api.deepseek.com'):
        keys(host, 'k-test')
    return keys


def _call(spec, monkeypatch, status, body):
    def urlopen(request, timeout):
        if status is None:
            raise urllib.error.URLError(TimeoutError('timed out'))
        raise urllib.error.HTTPError(request.full_url, status, 'x', {}, io.BytesIO(body.encode()))

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    client = connect(spec, post=lambda call: transport.post(call, attempts=2, sleep=lambda s: None))
    with pytest.raises(ProviderError) as caught:
        client.decide(DECIDE) if client.wire.respond is None else client.respond(RESPOND)
    return caught.value


@pytest.mark.parametrize('failure', _FAILURES, ids=lambda f: f'{f[0]}-{f[1]}-{f[3]}')
def test_each_vendor_failure_is_classified(every_key, monkeypatch, failure):
    spec, status, body, kind = failure
    assert _call(spec, monkeypatch, status, body).kind == kind


def test_every_shipped_wire_and_every_kind_is_covered():
    assert set(_SPECS.values()) == set(WIRES)
    for wire in WIRES:
        specs = [s for s, w in _SPECS.items() if w == wire]
        assert {'auth', 'timeout'} <= {k for s, _, _, k in _FAILURES if s in specs}, wire
    assert {k for *_, k in _FAILURES} == ERROR_KINDS


def test_the_planted_control_a_plain_client_error_carries_no_kind(every_key, monkeypatch):
    assert _call('deepseek:d', monkeypatch, 400, '{"error":{"message":"bad schema"}}').kind == ''


def test_a_timeout_does_not_move_to_the_next_slot_but_an_account_failure_does(every_key, monkeypatch):
    every_key('api.deepseek.com', 'k-two', username='api-key-2')
    assert 'timeout' not in ACCOUNT_KINDS
    assert 'slot' not in str(_call('deepseek:d', monkeypatch, None, ''))
    assert 'slot 2: exhausted' in str(_call('deepseek:d', monkeypatch, 402, '{}'))


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match='unknown kind'):
        ProviderError('x', kind='quota')
