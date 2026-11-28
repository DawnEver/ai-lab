"""``python -m ai_lab.serve ensure <provider>``: start a keyless local provider's server if it is down.

The catalogue row (``auth = "local"``) says WHERE the server answers (its ``url``); the user's
``serve.toml`` beside ``providers.toml`` says HOW to start it on this machine -- one table per provider
holding ``command``, the full argv minus ``--host``/``--port`` (derived from the row's url, so the two
cannot disagree). A machine path lives only there, never in the packaged catalogue::

    [llamacpp]
    command = ["~/tools/llama.cpp/llama-server", "-m", "~/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf",
               "--alias", "qwen3.8-27b", "--jinja", "-ngl", "99", "-c", "196608", "-fa", "on",
               "-ctk", "q8_0", "-ctv", "q8_0", "--parallel", "1"]

Several models behind one server: ``[<provider>.models.<name>]`` tables (and ``[<provider>.shared]`` for
what every model shares) hold each model's llama-server options, keys as the long option without ``--``.
They are the ONE source; the router preset llama-server reads (``--models-preset``, INI) is rendered from
them into ``serve-<provider>.ini`` on every ``ensure``, never hand-edited::

    [llamacpp]
    command = ["~/tools/llama.cpp/llama-server", "--models-max", "1"]
    default = "qwen3.8-27b"

    [llamacpp.shared]
    jinja = true

    [llamacpp.models."qwen3.8-27b"]
    model = "~/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
    ctx-size = 196608

``ensure --model <name>`` loads that model (the router swaps it in); ``models <provider>`` lists them.

A started server is detached; its log is ``serve-<provider>.log`` in the same directory. A launcher
passes ``--holder <its pid>`` to ``ensure``, and to ``release`` when its session ends: the server this
module started is stopped when its last LIVE holder lets go (a holder that died without releasing does not
keep it up). A server it did not start (one already answering, or ``ensure`` without a holder) is never
stopped. A launcher killed with its window never releases, so a server started for a holder is also
watched: a detached ``watch`` process checks the holders every :data:`WATCH_INTERVAL` seconds and stops
the server once none is alive. The state is ``serve-<provider>.pid`` and ``serve-<provider>.holders``
beside the log. Setup and sizing: ``docs-src/local-llama.md``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
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

__all__ = ['argv', 'ensure', 'models_of', 'preset', 'release', 'serve_file', 'specs', 'watch']

#: The status ``/health`` answers once the model is loaded.
HEALTHY = 200

#: Seconds a fresh server gets to load its weights and answer ``/health``.
TIMEOUT = 600.0

#: Seconds between two checks of the watcher for a live holder.
WATCH_INTERVAL = 5.0


def serve_file() -> Path:
    """Where the user's ``serve.toml`` lives (it need not exist)."""
    return config_root(APP) / 'serve.toml'


def specs(path: Path) -> dict[str, dict]:
    """The provider tables of *path*; ``{}`` when it is absent."""
    return tomllib.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def _home(value: str) -> str:
    return str(Path(value).expanduser()) if value.startswith('~') else value


def _ini(value: object) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return _home(value) if isinstance(value, str) else str(value)


def preset(spec: Mapping) -> str:
    """The router preset (INI) of *spec*'s ``shared`` and ``models`` tables, ``~`` expanded."""
    sections = [('*', spec.get('shared') or {}), *dict(spec.get('models') or {}).items()]
    lines = ['version = 1', '']
    for name, options in sections:
        if options or name != '*':
            lines += [f'[{name}]', *(f'{key} = {_ini(value)}' for key, value in options.items()), '']
    return '\n'.join(lines)


def models_of(spec: Mapping) -> tuple[list[str], str | None]:
    """*spec*'s model names, sorted, and its ``default`` (``None`` when it names none)."""
    names = sorted(spec.get('models') or {})
    default = spec.get('default')
    if default is not None and default not in names:
        msg = f'serve.toml default {default!r} names no model; the models are {names}'
        raise Unsupported(msg)
    return names, default


def argv(row: Provider, spec: Mapping, *, preset_file: Path | None = None) -> list[str]:
    """*spec*'s ``command`` with ``~`` expanded, its rendered preset, and ``--host``/``--port`` from *row*'s url."""
    command = [_home(part) for part in spec['command']]
    if clash := sorted({'--host', '--port', '--models-preset'} & set(command)):
        msg = f'serve.toml [{row.name}] names {clash[0]}; drop it, ai_lab.serve sets it'
        raise Unsupported(msg)
    if spec.get('models'):
        target = preset_file if preset_file is not None else serve_file().with_name(f'serve-{row.name}.ini')
        _write(target, preset(spec))
        command += ['--models-preset', str(target)]
    url = urlsplit(row.url)
    return [*command, '--host', url.hostname or '127.0.0.1', '--port', str(url.port or 80)]


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 -- catalogue url
        return json.loads(response.read())


def _load(url: str, model: str, *, timeout: float = TIMEOUT) -> None:
    """Have the router re-read its preset, load *model*, and wait until it is loaded."""
    base = url.rstrip('/')
    _get_json(f'{base}/models?reload=1')
    request = urllib.request.Request(  # noqa: S310 -- catalogue url
        f'{base}/models/load', data=json.dumps({'model': model}).encode(), headers={'content-type': 'application/json'}
    )
    with urllib.request.urlopen(request, timeout=30):  # noqa: S310 -- catalogue url
        pass
    waited = 0.0
    while waited < timeout:
        rows = {m.get('id'): m for m in _get_json(f'{base}/models').get('data', [])}
        if (rows.get(model, {}).get('status') or {}).get('value') == 'loaded':
            return
        time.sleep(1.0)
        waited += 1.0
    msg = f'model {model!r} not loaded after {timeout:.0f}s; see the server log beside serve.toml'
    raise Unsupported(msg)


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(f'{url.rstrip("/")}/health', timeout=2) as response:  # noqa: S310 -- catalogue url
            return response.status == HEALTHY
    except OSError:
        return False


def _watch_spawn(provider: str, log: Path) -> None:
    """The detached watcher that stops *provider*'s server once no holder is alive."""
    _spawn([sys.executable, '-m', 'ai_lab.serve', 'watch', provider], log)


def _spawn(command: list[str], log: Path) -> int:
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab') as sink:
        return subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=sink, stderr=sink, creationflags=flags,
            start_new_session=os.name != 'nt', close_fds=True,
        ).pid  # fmt: skip


#: Windows: a process handle that may only query, and the exit code of a process still running.
_QUERY_LIMITED, _STILL_ACTIVE = 0x1000, 259


def _alive(pid: int) -> bool:
    """Whether *pid* runs. Never ``os.kill(pid, 0)`` on Windows: there it TERMINATES the process."""
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(_QUERY_LIMITED, 0, pid)  # 0: the handle is not inherited
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == _STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def _stop(pid: int) -> None:
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(pid), '/F'], capture_output=True, check=False)  # noqa: S607
    else:
        os.kill(pid, 15)


def _state(row: Provider, state: Path | None) -> tuple[Path, Path]:
    directory = state if state is not None else serve_file().parent
    return directory / f'serve-{row.name}.pid', directory / f'serve-{row.name}.holders'


def _holders(path: Path) -> list[int]:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.with_name(f'{path.name}.new')
    fresh.write_text(text, encoding='utf-8')
    fresh.replace(path)


def ensure(
    row: Provider,
    spec: Mapping,
    *,
    probe: Callable[[str], bool] = _probe,
    spawn: Callable[[list[str]], int | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = TIMEOUT,
    holder: int | None = None,
    state: Path | None = None,
    watcher: Callable[[str, Path], None] = _watch_spawn,
    model: str | None = None,
    load: Callable[[str, str], None] = _load,
) -> str:
    """``'running'`` when *row*'s server answers, else start it and return ``'started'`` once it does.

    A *holder* (the launcher's pid) is recorded as holding the server, and a server started for it is
    recorded as this module's, so :func:`release` may stop it. A *model* of a router spec is loaded.
    """
    if row.auth != 'local':
        msg = f'{row.name} is auth {row.auth!r}; only a local provider has a server to start'
        raise Unsupported(msg)
    if model is not None and model not in (names := models_of(spec)[0]):
        msg = f'serve.toml [{row.name}] has no model {model!r}; the models are {names}'
        raise Unsupported(msg)
    pid_file, holders_file = _state(row, state)
    preset_file = pid_file.with_suffix('.ini')
    if probe(row.url):
        if holder is not None:
            _write(holders_file, json.dumps(sorted({*_holders(holders_file), holder})))
        if spec.get('models'):
            _write(preset_file, preset(spec))
        outcome = 'running'
    else:
        log = serve_file().with_name(f'serve-{row.name}.log')
        command = argv(row, spec, preset_file=preset_file)
        pid = _spawn(command, log) if spawn is None else spawn(command)
        _await(row, probe, sleep, timeout, log)
        if holder is not None:
            _write(holders_file, json.dumps([holder]))
            if pid:
                _write(pid_file, str(pid))
                watcher(row.name, log)
        outcome = 'started'
    if model is not None:
        load(row.url, model)
    return outcome


def _await(
    row: Provider, probe: Callable[[str], bool], sleep: Callable[[float], None], timeout: float, log: Path
) -> None:
    """Wait until *row*'s fresh server answers ``/health``; refuse after *timeout* naming its log."""
    waited = 0.0
    while waited < timeout:
        sleep(1.0)
        waited += 1.0
        if probe(row.url):
            return
    msg = f'{row.name} server at {row.url} not healthy after {timeout:.0f}s; see {log}'
    raise Unsupported(msg)


def release(
    row: Provider,
    holder: int,
    *,
    alive: Callable[[int], bool] = _alive,
    stop: Callable[[int], None] = _stop,
    state: Path | None = None,
) -> str:
    """Let *holder* go: ``'stopped'`` once the server this module started has no live holder left.

    Else ``'held'`` (another live session uses it) or ``'not-ours'`` (a server this module did not start).
    """
    pid_file, holders_file = _state(row, state)
    left = [pid for pid in _holders(holders_file) if pid != holder and alive(pid)]
    _write(holders_file, json.dumps(left))
    if not pid_file.is_file():
        return 'not-ours'
    if left:
        return 'held'
    server = int(pid_file.read_text(encoding='utf-8'))
    pid_file.unlink()
    if alive(server):
        stop(server)
    return 'stopped'


def watch(
    row: Provider,
    *,
    alive: Callable[[int], bool] = _alive,
    stop: Callable[[int], None] = _stop,
    sleep: Callable[[float], None] = time.sleep,
    state: Path | None = None,
    interval: float = WATCH_INTERVAL,
) -> str:
    """Stop the server this module started once none of its holders is alive: ``'stopped'``.

    Leaves with ``'released'`` once the server is no longer this module's (a :func:`release` stopped it).
    """
    pid_file, holders_file = _state(row, state)
    while True:
        sleep(interval)
        if not pid_file.is_file():
            return 'released'
        server = int(pid_file.read_text(encoding='utf-8'))
        if not alive(server):
            pid_file.unlink(missing_ok=True)
            return 'released'
        if not any(alive(pid) for pid in _holders(holders_file)):
            pid_file.unlink(missing_ok=True)
            _write(holders_file, '[]')
            stop(server)
            return 'stopped'


def main(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m ai_lab.serve', description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='verb', required=True)
    verbs = {
        'ensure': 'start the server unless it already answers',
        'release': 'let a holder go; stop the server this module started once no live holder is left',
        'watch': 'stop the server this module started once no holder is alive (spawned by ensure)',
        'models': "list the provider's models (serve.toml) and its default",
    }
    for verb, text in verbs.items():
        verb_parser = sub.add_parser(verb, help=text)
        verb_parser.add_argument('provider')
        if verb in {'ensure', 'release'}:
            verb_parser.add_argument('--holder', type=int, required=verb == 'release', help='the launcher pid')
        if verb == 'ensure':
            verb_parser.add_argument('--model', help='load this model of a router spec')
        if verb == 'models':
            verb_parser.add_argument('--json', action='store_true', help='print {"models": [...], "default": ...}')
    parsed = parser.parse_args(args)
    rows = catalogue()
    if parsed.provider not in rows:
        sys.stderr.write(f'unknown provider {parsed.provider!r}; the catalogue has {sorted(rows)}\n')
        return 2
    row = rows[parsed.provider]
    if parsed.verb in {'watch', 'release'}:
        out = watch(row) if parsed.verb == 'watch' else release(row, parsed.holder)
        sys.stdout.write(f'{parsed.provider}: {out}\n')
        return 0
    spec = specs(serve_file()).get(parsed.provider)
    if spec is None:
        sys.stderr.write(f'no [{parsed.provider}] in {serve_file()}; see docs-src/local-llama.md\n')
        return 2
    try:
        sys.stdout.write(_spec_verb(parsed, row, spec))
    except Unsupported as error:
        sys.stderr.write(f'{error}\n')
        return 1
    return 0


def _spec_verb(parsed: argparse.Namespace, row: Provider, spec: Mapping) -> str:
    """The output of ``models`` or ``ensure`` on *row* with its serve.toml *spec*."""
    if parsed.verb == 'models':
        names, default = models_of(spec)
        if parsed.json:
            return json.dumps({'models': names, 'default': default}) + '\n'
        return ''.join(f'{"*" if name == default else " "} {name}\n' for name in names)
    return f'{parsed.provider}: {ensure(row, spec, holder=parsed.holder, model=parsed.model)}\n'


if __name__ == '__main__':
    sys.exit(main())
