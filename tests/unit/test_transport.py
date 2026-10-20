import io
import urllib.error

import pytest

from ai_lab import ProviderError
from ai_lab.transport import HttpCall, post


def _http_error(code, body=b'detail'):
    return urllib.error.HTTPError('u', code, 'x', {}, io.BytesIO(body))


def test_a_retryable_status_is_retried_then_succeeds(monkeypatch):
    replies = [_http_error(429), _http_error(503)]

    class Ok(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(request, timeout):
        if replies:
            raise replies.pop(0)
        return Ok(b'{"ok": true}')

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    sleeps = []
    assert post(HttpCall('https://x.test', {}), sleep=sleeps.append) == {'ok': True}
    assert sleeps == [1.0, 2.0]


def test_a_client_error_is_not_retried(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(1)
        raise _http_error(401)

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    with pytest.raises(ProviderError, match='HTTP 401') as caught:
        post(HttpCall('https://x.test', {}), sleep=lambda s: None)
    assert len(calls) == 1
    assert caught.value.kind == 'auth'


def test_retries_are_bounded(monkeypatch):
    def urlopen(request, timeout):
        msg = 'down'
        raise urllib.error.URLError(msg)

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    with pytest.raises(ProviderError, match='after 3 attempts') as caught:
        post(HttpCall('https://x.test', {}), sleep=lambda s: None)
    assert caught.value.kind == ''


def test_a_rate_limit_past_its_retries_is_an_account_failure(monkeypatch):
    def urlopen(request, timeout):
        raise _http_error(429, b'slow down')

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    with pytest.raises(ProviderError, match='after 3 attempts') as caught:
        post(HttpCall('https://x.test', {}), sleep=lambda s: None)
    assert caught.value.kind == 'rate_limit'


def test_an_exhausted_quota_is_not_retried(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(1)
        raise _http_error(429, b'{"error":{"type":"insufficient_quota","code":"credit_balance_exhausted"}}')

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    with pytest.raises(ProviderError, match='insufficient_quota') as caught:
        post(HttpCall('https://x.test', {}), sleep=lambda s: None)
    assert len(calls) == 1
    assert caught.value.kind == 'exhausted'


class _Body(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_200_carrying_only_an_error_is_retried(monkeypatch):
    """DeepSeek answers an overloaded queue with HTTP 200 and ``{"error": ...}`` -- a transient failure."""
    replies = [b'{"error": {"message": "unable to start processing within the 900-second timeout"}}', b'{"ok": true}']
    monkeypatch.setattr('urllib.request.urlopen', lambda request, timeout: _Body(replies.pop(0)))
    sleeps = []
    assert post(HttpCall('https://x.test', {}), sleep=sleeps.append) == {'ok': True}
    assert sleeps == [1.0]


def test_an_exhausted_balance_in_a_200_body_is_not_retried(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(1)
        return _Body(b'{"error": {"code": "insufficient_quota"}}')

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    with pytest.raises(ProviderError, match='insufficient_quota'):
        post(HttpCall('https://x.test', {}), sleep=lambda s: None)
    assert len(calls) == 1
