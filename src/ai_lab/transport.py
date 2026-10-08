"""How bytes reach a model: an HTTPS POST, or a local CLI fed on stdin. Nothing vendor-specific.

A wire builds a :class:`HttpCall` or :class:`CliCall` and parses what comes back; this module
only executes. Secrets travel in ``headers``, which never appear in a repr.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ai_lab.errors import ProviderError, Unsupported

__all__ = ['CliCall', 'CliResult', 'HttpCall', 'post', 'run']

#: Statuses worth another attempt: rate limit, overload, transient server failure.
_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


@dataclass(frozen=True, slots=True)
class HttpCall:
    url: str
    body: Mapping
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class CliCall:
    """``argv[0]`` is the binary NAME; ``files`` are written into a fresh empty working directory
    and ``{dir}`` in any argument is replaced by its path. ``output`` names the file read back
    (``None`` reads stdout)."""

    argv: tuple[str, ...]
    stdin: str
    files: Mapping[str, bytes] = field(default_factory=dict)
    output: str | None = None


@dataclass(frozen=True, slots=True)
class CliResult:
    text: str


def post(call: HttpCall, *, timeout_s: float = 120.0, attempts: int = 3, sleep: Callable = time.sleep) -> dict:
    """POST JSON, retrying a retryable status or a network failure ``attempts`` times with backoff."""
    data = json.dumps(call.body).encode('utf-8')
    last = ''
    for attempt in range(attempts):
        request = urllib.request.Request(
            call.url, data=data, method='POST', headers={'Content-Type': 'application/json', **call.headers}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as reply:
                return json.loads(reply.read().decode('utf-8'))
        except urllib.error.HTTPError as error:
            detail = error.read().decode('utf-8', 'replace')[:500]
            last = f'HTTP {error.code} from {call.url}: {detail}'
            if error.code not in _RETRYABLE:
                raise ProviderError(last) from None
        except (urllib.error.URLError, TimeoutError) as error:
            last = f'{call.url} unreachable: {error}'
        if attempt + 1 < attempts:
            sleep(2.0**attempt)
    raise ProviderError(f'{last} (after {attempts} attempts)')


def run(call: CliCall, *, timeout_s: float = 600.0) -> CliResult:
    """Run a local CLI in an empty temporary directory, prompt on stdin, no shell."""
    binary = shutil.which(call.argv[0])
    if binary is None:
        raise Unsupported(f'{call.argv[0]!r} is not on PATH; install it or point the provider at its binary')
    with tempfile.TemporaryDirectory(prefix='ai_lab_') as tmp:
        for name, content in call.files.items():
            (Path(tmp) / name).write_bytes(content)
        argv = [binary, *(a.replace('{dir}', tmp) for a in call.argv[1:])]
        try:
            done = subprocess.run(
                argv, input=call.stdin, capture_output=True, text=True, encoding='utf-8', cwd=tmp, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            raise ProviderError(f'{call.argv[0]} did not finish within {timeout_s} s') from None
        if done.returncode != 0:
            raise ProviderError(f'{call.argv[0]} exited {done.returncode}: {(done.stderr or done.stdout)[-800:]}')
        if call.output is None:
            return CliResult(done.stdout)
        out = Path(tmp) / call.output
        if not out.exists():
            raise ProviderError(f'{call.argv[0]} wrote no {call.output}; stdout tail: {done.stdout[-400:]}')
        return CliResult(out.read_text(encoding='utf-8'))
