"""``ai_lab.serve``: a keyless local provider's server is started on demand, from the user's ``serve.toml``."""

from __future__ import annotations

import pytest

from ai_lab.client import Provider
from ai_lab.errors import Unsupported
from ai_lab.serve import argv, ensure, release, specs, watch

ROW = Provider(name='llamacpp', wire='openai_chat', url='http://127.0.0.1:8080', auth='local', openai_path='/v1')


def test_argv_appends_host_and_port_from_the_catalogue_url():
    assert argv(ROW, {'command': ['llama-server', '-m', 'm.gguf']}) == [
        'llama-server', '-m', 'm.gguf', '--host', '127.0.0.1', '--port', '8080',
    ]  # fmt: skip


def test_argv_refuses_a_command_that_names_its_own_host_or_port():
    with pytest.raises(Unsupported, match='--port'):
        argv(ROW, {'command': ['llama-server', '--port', '9000']})


def test_a_running_server_is_not_started_again():
    spawned = []
    assert ensure(ROW, {'command': ['x']}, probe=lambda _: True, spawn=lambda c: spawned.append(c) or 1) == 'running'
    assert spawned == []


def test_a_down_server_is_spawned_and_waited_for():
    state = iter([False, False, True])
    spawned = []
    out = ensure(
        ROW,
        {'command': ['x']},
        probe=lambda _: next(state),
        spawn=lambda c: spawned.append(c) or 1,
        sleep=lambda _: None,
    )
    assert out == 'started'
    assert spawned == [['x', '--host', '127.0.0.1', '--port', '8080']]


def test_a_server_that_never_answers_is_reported_with_its_log():
    with pytest.raises(Unsupported, match='not healthy'):
        ensure(ROW, {'command': ['x']}, probe=lambda _: False, spawn=lambda _: None, sleep=lambda _: None, timeout=3)


def test_an_api_row_is_refused():
    row = Provider(name='p', wire='openai_chat', url='https://p.example', auth='api', openai_path='/v1')
    with pytest.raises(Unsupported, match='local'):
        ensure(row, {'command': ['x']}, probe=lambda _: True, spawn=lambda _: None)


def test_specs_names_the_file_when_a_provider_has_none(tmp_path):
    path = tmp_path / 'serve.toml'
    path.write_text('[other]\ncommand = ["x"]\n', encoding='utf-8')
    assert specs(path) == {'other': {'command': ['x']}}
    assert specs(tmp_path / 'absent.toml') == {}


# A server ai_lab.serve started is held by the sessions that ensured it and stopped when the last lets go.


def _started(tmp_path, holder):
    answers = iter([False, True])
    return ensure(
        ROW, {'command': ['x']}, probe=lambda _: next(answers), spawn=lambda _: 4242,
        sleep=lambda _: None, holder=holder, state=tmp_path, watcher=lambda *_: None,
    )  # fmt: skip


def test_the_last_holder_to_release_stops_the_server_it_started(tmp_path):
    assert _started(tmp_path, 100) == 'started'
    assert ensure(ROW, {'command': ['x']}, probe=lambda _: True, holder=200, state=tmp_path) == 'running'
    stopped = []
    alive = {100, 200, 4242}.__contains__
    assert release(ROW, 100, alive=alive, stop=stopped.append, state=tmp_path) == 'held'
    assert stopped == []
    assert release(ROW, 200, alive=alive, stop=stopped.append, state=tmp_path) == 'stopped'
    assert stopped == [4242]


def test_a_dead_holder_never_keeps_the_server_up(tmp_path):
    _started(tmp_path, 100)
    ensure(ROW, {'command': ['x']}, probe=lambda _: True, holder=200, state=tmp_path)
    stopped = []
    assert release(ROW, 100, alive={100, 4242}.__contains__, stop=stopped.append, state=tmp_path) == 'stopped'
    assert stopped == [4242], 'holder 200 died without releasing'


def test_a_server_serve_did_not_start_is_never_stopped(tmp_path):
    ensure(ROW, {'command': ['x']}, probe=lambda _: True, holder=100, state=tmp_path)
    stopped = []
    assert release(ROW, 100, alive=lambda _: True, stop=stopped.append, state=tmp_path) == 'not-ours'
    assert stopped == []


def test_the_watcher_stops_the_server_once_every_holder_is_gone_even_without_release(tmp_path):
    # A closed terminal kills the launcher before it can release: the watcher sees no live holder and stops.
    _started(tmp_path, 100)
    living = {100, 4242}
    ticks = []

    def sleep(_):
        ticks.append(1)
        if len(ticks) == 2:
            living.discard(100)  # the window is closed

    stopped = []
    assert watch(ROW, alive=living.__contains__, stop=stopped.append, sleep=sleep, state=tmp_path) == 'stopped'
    assert stopped == [4242]
    assert not (tmp_path / 'serve-llamacpp.pid').exists()


def test_the_watcher_leaves_when_the_server_is_released_or_gone(tmp_path):
    _started(tmp_path, 100)
    release(ROW, 100, alive=lambda _: True, stop=lambda _: None, state=tmp_path)
    assert watch(ROW, alive=lambda _: True, stop=lambda _: None, sleep=lambda _: None, state=tmp_path) == 'released'
