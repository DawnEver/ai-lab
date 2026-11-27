"""``python -m ai_lab.serve ensure <provider>``: start a keyless local provider's server if it is down.

The catalogue row (``auth = "local"``) says WHERE the server answers (its ``url``); the user's
``serve.toml`` beside ``providers.toml`` says HOW to start it on this machine -- one table per provider
holding ``command``, the full argv minus ``--host``/``--port`` (derived from the row's url, so the two
cannot disagree). A machine path lives only there, never in the packaged catalogue::

    [llamacpp]
    command = ["~/tools/llama.cpp/llama-server", "-m", "~/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf",
               "--alias", "qwen3.8-27b", "--jinja", "-ngl", "99", "-c", "196608", "-fa", "on",
               "-ctk", "q8_0", "-ctv", "q8_0", "--parallel", "1"]

A started server is detached and outlives the caller; its log is ``serve-<provider>.log`` in the same
directory. Setup and sizing: ``docs-src/local-llama.md``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tomllib
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from lab_commons.paths import config_root

from ai_lab.client import APP, Provider, catalogue
from ai_lab.errors import Unsupported

__all__ = ['argv', 'ensure', 'serve_file', 'specs']

#: The status ``/health`` answers once the model is loaded.
HEALTHY = 200

#: Seconds a fresh server gets to load its weights and answer ``/health``.
TIMEOUT = 600.0


def serve_file() -> Path:
    """Where the user's ``serve.toml`` lives (it need not exist)."""
    return config_root(APP) / 'serve.toml'


def specs(path: Path) -> dict[str, dict]:
    """The provider tables of *path*; ``{}`` when it is absent."""
    return tomllib.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def argv(row: Provider, spec: Mapping) -> list[str]:
    """*spec*'s ``command`` with ``~`` expanded and ``--host``/``--port`` taken from *row*'s url."""
    command = [str(Path(part).expanduser()) if part.startswith('~') else part for part in spec['command']]
    if clash := sorted({'--host', '--port'} & set(command)):
        msg = f'serve.toml [{row.name}] names {clash[0]}; drop it, the catalogue url {row.url} sets it'
        raise Unsupported(msg)
    url = urlsplit(row.url)
    return [*command, '--host', url.hostname or '127.0.0.1', '--port', str(url.port or 80)]


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(f'{url.rstrip("/")}/health', timeout=2) as response:  # noqa: S310 -- catalogue url
            return response.status == HEALTHY
    except OSError:
        return False


def _spawn(command: list[str], log: Path) -> None:
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    with log.open('ab') as sink:
        subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=sink, stderr=sink, creationflags=flags,
            start_new_session=os.name != 'nt', close_fds=True,
        )  # fmt: skip


def ensure(
    row: Provider,
    spec: Mapping,
    *,
    probe: Callable[[str], bool] = _probe,
    spawn: Callable[[list[str]], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = TIMEOUT,
) -> str:
    """``'running'`` when *row*'s server answers, else start it and return ``'started'`` once it does."""
    if row.auth != 'local':
        msg = f'{row.name} is auth {row.auth!r}; only a local provider has a server to start'
        raise Unsupported(msg)
    if probe(row.url):
        return 'running'
    log = serve_file().with_name(f'serve-{row.name}.log')
    command = argv(row, spec)
    if spawn is None:
        log.parent.mkdir(parents=True, exist_ok=True)
        _spawn(command, log)
    else:
        spawn(command)
    waited = 0.0
    while waited < timeout:
        sleep(1.0)
        waited += 1.0
        if probe(row.url):
            return 'started'
    msg = f'{row.name} server at {row.url} not healthy after {timeout:.0f}s; see {log}'
    raise Unsupported(msg)


def main(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m ai_lab.serve', description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='verb', required=True)
    sub.add_parser('ensure', help='start the server unless it already answers').add_argument('provider')
    parsed = parser.parse_args(args)
    rows = catalogue()
    if parsed.provider not in rows:
        sys.stderr.write(f'unknown provider {parsed.provider!r}; the catalogue has {sorted(rows)}\n')
        return 2
    spec = specs(serve_file()).get(parsed.provider)
    if spec is None:
        sys.stderr.write(f'no [{parsed.provider}] in {serve_file()}; see docs-src/local-llama.md\n')
        return 2
    try:
        sys.stdout.write(f'{parsed.provider}: {ensure(rows[parsed.provider], spec)}\n')
    except Unsupported as error:
        sys.stderr.write(f'{error}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
