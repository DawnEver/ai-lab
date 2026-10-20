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
#: A 429 whose body says the account is out of money is not transient: waiting will not refill it.
_EXHAUSTED = ('insufficient_quota', 'credit_balance_exhausted')
#: Statuses that say THIS KEY may not call: another key of the account may.
_AUTH = frozenset({401, 403})
_RATE_LIMIT = 429


@dataclass(frozen=True, slots=True)
class HttpCall:
    """An HTTPS POST: URL, JSON body, and headers that never appear in a repr."""

    url: str
    body: Mapping
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


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
            if any(marker in detail for marker in _EXHAUSTED):
                raise ProviderError(last, kind='exhausted') from None
            if error.code not in _RETRYABLE:
                raise ProviderError(last, kind='auth' if error.code in _AUTH else '') from None
            last_kind = 'rate_limit' if error.code == _RATE_LIMIT else ''
        except (urllib.error.URLError, TimeoutError) as error:
            last = f'{call.url} unreachable: {error}'
            last_kind = ''
        if attempt + 1 < attempts:
            sleep(2.0**attempt)
    msg = f'{last} (after {attempts} attempts)'
    raise ProviderError(msg, kind=last_kind)
