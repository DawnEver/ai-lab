"""providers.toml is the ONE provider catalogue: a row's url, a path per protocol it serves, and its role models.

One vendor account is one row. A second protocol into the same account (DeepSeek's Anthropic-compatible
endpoint beside its OpenAI-compatible one) is a ``<protocol>_path`` of that row, and the wire a call uses
is picked by protocol -- never a second provider.
"""

import pytest

from ai_lab import Provider, ResponseRequest, Text, Unsupported, connect, providers
from ai_lab.client import PROTOCOLS, ROLES
from ai_lab.wire import WIRES

REQ = ResponseRequest('i', (Text('x'),), {'type': 'object'})


def _post(payload):
    calls = []

    def post(call, **_):
        calls.append(call)
        return payload

    post.calls = calls
    return post


def test_every_wire_declares_a_known_protocol():
    assert {w.protocol for w in WIRES.values()} == PROTOCOLS


def test_deepseek_is_one_row_serving_two_protocols():
    row = providers()['deepseek']
    assert row.url == 'https://api.deepseek.com'
    assert row.endpoint('openai') == 'https://api.deepseek.com/v1'
    assert row.endpoint('anthropic') == 'https://api.deepseek.com/anthropic'
    assert row.protocols == frozenset({'openai', 'anthropic'})
    assert not any(name != 'deepseek' and r.host == row.host for name, r in providers().items())


def test_the_claude_role_models_are_catalogue_data():
    assert providers()['deepseek'].models == {
        'base': 'deepseek-v4-flash[1m]',
        'fable': 'deepseek-v4-pro[1m]',
        'opus': 'deepseek-v4-flash[1m]',
        'sonnet': 'deepseek-v4-flash[1m]',
        'haiku': 'deepseek-v4-flash[1m]',
        'subagent': 'deepseek-v4-flash[1m]',
    }
    assert frozenset({'base', 'fable', 'opus', 'sonnet', 'haiku', 'subagent', 'codex'}) == ROLES


def test_the_wire_is_picked_by_protocol(credential_store):
    credential_store('api.deepseek.com', 'dk-test')
    reply = {'choices': [{'message': {'content': '{}'}}]}
    default = _post(reply)
    connect('deepseek:deepseek-v4-flash', post=default).respond(REQ)
    assert default.calls[0].url == 'https://api.deepseek.com/v1/chat/completions'
    anthropic = _post({'content': [{'type': 'text', 'text': '{}'}]})
    client = connect('deepseek:deepseek-v4-flash', protocol='anthropic', post=anthropic)
    assert client.wire.name == 'anthropic_messages'
    client.respond(REQ)
    assert anthropic.calls[0].url == 'https://api.deepseek.com/anthropic/messages'


def test_a_protocol_the_row_does_not_serve_is_refused_naming_the_ones_it_does():
    with pytest.raises(Unsupported, match=r"openai does not serve protocol 'anthropic'.*\['openai'\]"):
        connect('openai:gpt-x', protocol='anthropic')


def test_an_unknown_protocol_is_refused():
    with pytest.raises(Unsupported, match='unknown protocol'):
        connect('deepseek:m', protocol='smoke')


def test_a_row_whose_wire_has_no_path_for_its_protocol_is_refused():
    with pytest.raises(Unsupported, match=r"wire 'anthropic_messages' speaks 'anthropic'.*anthropic_path"):
        Provider('p', 'anthropic_messages', 'https://x.test', 'api', key_name='K', openai_path='/v1')


def test_a_url_carrying_a_path_is_refused():
    with pytest.raises(Unsupported, match='origin'):
        Provider('p', 'openai_chat', 'https://x.test/v1', 'local', openai_path='/v1')


def test_a_model_map_with_an_unknown_role_or_no_base_is_refused():
    with pytest.raises(Unsupported, match="unknown role 'tiny'"):
        Provider('p', 'openai_chat', 'https://x.test', 'local', openai_path='/v1', models={'base': 'm', 'tiny': 't'})
    with pytest.raises(Unsupported, match="needs 'base'"):
        Provider('p', 'openai_chat', 'https://x.test', 'local', openai_path='/v1', models={'opus': 'm'})


def test_the_planted_control_a_complete_row_loads():
    row = Provider('p', 'openai_chat', 'https://x.test', 'local', openai_path='/v1', models={'base': 'm'})
    assert row.endpoint('openai') == 'https://x.test/v1'
