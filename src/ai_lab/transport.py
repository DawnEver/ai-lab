"""How bytes reach a model: an HTTPS POST with a ceiling and a classified failure. Nothing vendor-specific.

A wire builds a :class:`HttpCall` and parses what comes back; this module only executes. Secrets travel
in ``headers``, which never appear in a repr. Every wait has a ceiling (:data:`HTTP_CEILING_S`).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ai_lab.errors import ProviderError, Unsupported

__all__ = ['HTTP_CEILING_S', 'HttpCall', 'post']

#: The longest one HTTP attempt may take, in seconds.
HTTP_CEILING_S = 120.0

#: Statuses worth another attempt: rate limit, overload, transient server failure.
_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
#: A body saying the account is out of money, whatever its status: waiting will not refill it. One
#: marker per vendor spelling (OpenAI, Anthropic, DeepSeek, OpenRouter, DashScope).
_EXHAUSTED = (
    'insufficient_quota', 'credit_balance_exhausted', 'credit balance is too low', 'Insufficient Balance',
    'Insufficient credits', 'Arrearage',
)  # fmt: skip
#: 402 Payment Required is exhaustion by status.
_PAYMENT = 402
#: Statuses that say THIS KEY may not call: another key of the account may.
_AUTH = frozenset({401, 403})
#: A body naming an invalid key under a generic status (Gemini answers a bad key with 400).
_AUTH_MARKERS = ('API_KEY_INVALID',)
_RATE_LIMIT = 429


@dataclass(frozen=True, slots=True)
class HttpCall:
    """An HTTPS POST: URL, JSON body, and headers that never appear in a repr."""

    url: str
    body: Mapping
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


def _final_kind(status: int, detail: str) -> str | None:
    """The kind of a failure no retry will fix; ``None`` for a status worth another attempt."""
    if status == _PAYMENT or any(marker in detail for marker in _EXHAUSTED):
        return 'exhausted'
    if status in _AUTH or any(marker in detail for marker in _AUTH_MARKERS):
        return 'auth'
    return None if status in _RETRYABLE else ''


def post(call: HttpCall, *, timeout_s: float = HTTP_CEILING_S, attempts: int = 3, sleep: Callable = time.sleep) -> dict:
    """POST JSON, retrying a retryable status or a network failure ``attempts`` times with backoff."""
    if not call.url.startswith(('https://', 'http://')):
        msg = f'{call.url!r} is not an http(s) URL; a provider endpoint is a web address'
        raise Unsupported(msg)
    data = json.dumps(call.body).encode('utf-8')
    last, last_kind = '', ''
    for attempt in range(attempts):
        request = urllib.request.Request(  # noqa: S310 -- the scheme is checked above
            call.url, data=data, method='POST', headers={'Content-Type': 'application/json', **call.headers}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as reply:  # noqa: S310 -- checked above
                body = json.loads(reply.read().decode('utf-8'))
            # A 200 carrying only an error (DeepSeek's overloaded queue) is a failed call, judged as a status.
            if not (isinstance(body, dict) and 'error' in body and 'choices' not in body):
                return body
            last = f'error body from {call.url}: {str(body["error"])[:500]}'
            if any(marker in last for marker in _EXHAUSTED):
                raise ProviderError(last, kind='exhausted')
        except urllib.error.HTTPError as error:
            detail = error.read().decode('utf-8', 'replace')[:500]
            last = f'HTTP {error.code} from {call.url}: {detail}'
            if (final := _final_kind(error.code, detail)) is not None:
                raise ProviderError(last, kind=final) from None
            last_kind = 'rate_limit' if error.code == _RATE_LIMIT else ''
        except (urllib.error.URLError, TimeoutError) as error:
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            timed_out = isinstance(reason, TimeoutError)
            last = f'{call.url} {"timed out" if timed_out else "unreachable"}: {error}'
            last_kind = 'timeout' if timed_out else ''
        if attempt + 1 < attempts:
            sleep(2.0**attempt)
    msg = f'{last} (after {attempts} attempts)'
    raise ProviderError(msg, kind=last_kind)
