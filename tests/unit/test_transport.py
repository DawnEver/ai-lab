import io
import sys
import urllib.error

import pytest
from lab_commons.proc import pid_alive

from ai_lab import ProviderError, Unsupported
from ai_lab.transport import CliCall, HttpCall, post, run


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
    with pytest.raises(ProviderError, match='HTTP 401'):
        post(HttpCall('https://x.test', {}), sleep=lambda s: None)
    assert len(calls) == 1


def test_retries_are_bounded(monkeypatch):
    def urlopen(request, timeout):
        msg = 'down'
        raise urllib.error.URLError(msg)

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    with pytest.raises(ProviderError, match='after 3 attempts'):
        post(HttpCall('https://x.test', {}), sleep=lambda s: None)


def test_a_missing_binary_is_unsupported():
    with pytest.raises(Unsupported, match='not on PATH'):
        run(CliCall(('no-such-binary-ai-lab',), ''))


_PARENT = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
open(sys.argv[1], 'w').write(str(child.pid))
time.sleep(60)
"""


def test_a_cli_past_its_ceiling_is_ended_as_a_tree(tmp_path):
    pid_file = tmp_path / 'child.pid'
    call = CliCall((sys.executable, '-c', _PARENT, str(pid_file)), '')
    with pytest.raises(ProviderError, match='ended its process tree'):
        run(call, timeout_s=3.0)
    child = int(pid_file.read_text())
    assert not pid_alive(child)


def test_an_exhausted_quota_is_not_retried(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(1)
        raise _http_error(429, b'{"error":{"type":"insufficient_quota","code":"credit_balance_exhausted"}}')

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    with pytest.raises(ProviderError, match='insufficient_quota'):
        post(HttpCall('https://x.test', {}), sleep=lambda s: None)
    assert len(calls) == 1
