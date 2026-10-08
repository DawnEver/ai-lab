"""How bytes reach a model: an HTTPS POST, or a local CLI fed on stdin. Nothing vendor-specific.

A wire builds a :class:`HttpCall` or :class:`CliCall` and parses what comes back; this module only
executes. Secrets travel in ``headers``, which never appear in a repr.

EVERY WAIT HAS A CEILING, AND A CLI THAT OUTLIVES ITS CEILING IS ENDED AS A TREE. ``claude`` and
``codex`` start children of their own, and ``subprocess.run``'s timeout kills only the process it
started -- the children are orphaned and keep a model session (and a core) busy. So the CLI is run
with :class:`subprocess.Popen` and, past :data:`CLI_CEILING_S`, ended by
:func:`lab_commons.proc.kill_process_tree` from its root pid, descendants first.
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

from lab_commons.proc import kill_process_tree

from ai_lab.errors import ProviderError, Unsupported

__all__ = ['CLI_CEILING_S', 'HTTP_CEILING_S', 'CliCall', 'CliResult', 'HttpCall', 'post', 'run']

#: The longest one HTTP attempt may take, and the longest a local CLI may run, in seconds.
HTTP_CEILING_S = 120.0
CLI_CEILING_S = 600.0
#: How long a killed tree's pipes may take to close.
_REAP_CEILING_S = 10.0

#: Statuses worth another attempt: rate limit, overload, transient server failure.
_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


@dataclass(frozen=True, slots=True)
class HttpCall:
    """An HTTPS POST: URL, JSON body, and headers that never appear in a repr."""

    url: str
    body: Mapping
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class CliCall:
    """A local CLI invocation, run in a fresh empty directory with the prompt on stdin.

    ``argv[0]`` is the binary NAME; ``files`` are written into a fresh empty working directory
    and ``{dir}`` in any argument is replaced by its path. ``output`` names the file read back
    (``None`` reads stdout).
    """

    argv: tuple[str, ...]
    stdin: str
    files: Mapping[str, bytes] = field(default_factory=dict)
    output: str | None = None


@dataclass(frozen=True, slots=True)
class CliResult:
    """What a CLI answered: its stdout, or the file it was told to write."""

    text: str


def post(call: HttpCall, *, timeout_s: float = HTTP_CEILING_S, attempts: int = 3, sleep: Callable = time.sleep) -> dict:
    """POST JSON, retrying a retryable status or a network failure ``attempts`` times with backoff."""
    if not call.url.startswith(('https://', 'http://')):
        msg = f'{call.url!r} is not an http(s) URL; a provider endpoint is a web address'
        raise Unsupported(msg)
    data = json.dumps(call.body).encode('utf-8')
    last = ''
    for attempt in range(attempts):
        request = urllib.request.Request(  # noqa: S310 -- the scheme is checked above
            call.url, data=data, method='POST', headers={'Content-Type': 'application/json', **call.headers}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as reply:  # noqa: S310 -- checked above
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
    msg = f'{last} (after {attempts} attempts)'
    raise ProviderError(msg)


def run(call: CliCall, *, timeout_s: float = CLI_CEILING_S) -> CliResult:
    """Run a local CLI in an empty temporary directory, prompt on stdin, no shell; end its TREE on timeout."""
    binary = shutil.which(call.argv[0])
    if binary is None:
        msg = f'{call.argv[0]!r} is not on PATH; install it or point the provider at its binary'
        raise Unsupported(msg)
    with tempfile.TemporaryDirectory(prefix='ai_lab_') as tmp:
        for name, content in call.files.items():
            (Path(tmp) / name).write_bytes(content)
        argv = [binary, *(a.replace('{dir}', tmp) for a in call.argv[1:])]
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            cwd=tmp,
        )
        try:
            stdout, stderr = process.communicate(call.stdin, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            ended = kill_process_tree(process.pid)
            process.communicate(timeout=_REAP_CEILING_S)
            msg = f'{call.argv[0]} did not finish within {timeout_s} s; ended its process tree ({len(ended)} pid(s))'
            raise ProviderError(msg) from None
        if process.returncode != 0:
            msg = f'{call.argv[0]} exited {process.returncode}: {(stderr or stdout)[-800:]}'
            raise ProviderError(msg)
        if call.output is None:
            return CliResult(stdout)
        out = Path(tmp) / call.output
        if not out.exists():
            msg = f'{call.argv[0]} wrote no {call.output}; stdout tail: {stdout[-400:]}'
            raise ProviderError(msg)
        return CliResult(out.read_text(encoding='utf-8'))
