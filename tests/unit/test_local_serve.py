"""``ai_lab.serve``: a keyless local provider's server is started on demand, from the user's ``serve.toml``."""

from __future__ import annotations

import pytest

from ai_lab.client import Provider
from ai_lab.errors import Unsupported
from ai_lab.serve import argv, ensure, specs

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
    assert ensure(ROW, {'command': ['x']}, probe=lambda _: True, spawn=spawned.append) == 'running'
    assert spawned == []


def test_a_down_server_is_spawned_and_waited_for():
    state = iter([False, False, True])
    spawned = []
    out = ensure(ROW, {'command': ['x']}, probe=lambda _: next(state), spawn=spawned.append, sleep=lambda _: None)
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
