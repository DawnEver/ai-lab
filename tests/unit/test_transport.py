import io
import urllib.error

import pytest

from ai_lab import ProviderError, Unsupported
from ai_lab.transport import CliCall, HttpCall, post, run


def _http_error(code):
    return urllib.error.HTTPError('u', code, 'x', {}, io.BytesIO(b'detail'))


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
    with pytest.raises(ProviderError, match='HTTP 401'):
        post(HttpCall('https://x.test', {}), sleep=lambda s: None)
    assert len(calls) == 1


def test_retries_are_bounded(monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.URLError('down')

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    with pytest.raises(ProviderError, match='after 3 attempts'):
        post(HttpCall('https://x.test', {}), sleep=lambda s: None)


def test_a_missing_binary_is_unsupported():
    with pytest.raises(Unsupported, match='not on PATH'):
        run(CliCall(('no-such-binary-ai-lab',), ''))
