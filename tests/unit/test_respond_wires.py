"""Every schema-responding wire: the schema reaches the vendor, images ride in its format, the
reply parses back to the object, and a refusal or malformed reply is a ProviderError."""

import base64
import json

import pytest

from ai_lab import Image, ProviderError, ResponseRequest, Text
from ai_lab.wire import anthropic_messages, gemini, openai_chat, openai_chat_json, openai_responses
from tests.conftest import PNG

SCHEMA = {
    'type': 'object',
    'properties': {'ok': {'type': 'boolean'}},
    'required': ['ok'],
    'additionalProperties': False,
}
REQ = ResponseRequest('Be brief.', (Text('hello'), Image(PNG)), SCHEMA, name='probe')
B64 = base64.b64encode(PNG).decode()


def test_openai_responses():
    call = openai_responses.WIRE.respond(REQ, 'gpt-x', 'https://api.openai.com/v1', 'k')
    assert call.url.endswith('/responses')
    fmt = call.body['text']['format']
    assert fmt == {'type': 'json_schema', 'name': 'probe', 'schema': SCHEMA, 'strict': True}
    assert call.body['input'][0]['content'][1]['image_url'] == f'data:image/png;base64,{B64}'
    raw = {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"ok": true}'}]}]}
    assert openai_responses.WIRE.parse_response(raw) == {'ok': True}
    with pytest.raises(ProviderError, match='refused'):
        openai_responses.WIRE.parse_response(
            {'output': [{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'no'}]}]}
        )


def test_an_open_schema_is_sent_non_strict():
    open_req = ResponseRequest('i', (Text('t'),), {'type': 'object'})
    assert openai_responses.WIRE.respond(open_req, 'm', 'e', 'k').body['text']['format']['strict'] is False


def test_openai_chat():
    call = openai_chat.WIRE.respond(REQ, 'deepseek-chat', 'https://api.deepseek.com/v1', 'k')
    assert call.url.endswith('/chat/completions')
    assert call.body['response_format']['json_schema']['schema'] == SCHEMA
    assert call.body['messages'][1]['content'][1] == {
        'type': 'image_url',
        'image_url': {'url': f'data:image/png;base64,{B64}'},
    }
    assert openai_chat.WIRE.parse_response({'choices': [{'message': {'content': '{"ok": false}'}}]}) == {'ok': False}
    with pytest.raises(ProviderError, match='not the declared JSON'):
        openai_chat.WIRE.parse_response({'choices': [{'message': {'content': 'sure!'}}]})


def test_openai_chat_without_a_key_sends_no_auth_header():
    assert openai_chat.WIRE.respond(REQ, 'llama', 'http://localhost:11434/v1', '').headers == {}


def test_anthropic_messages_marks_the_block_before_the_newest_as_a_cache_breakpoint():
    # An append-only context grows by one block per call; the automatic breakpoint sits on the LAST
    # block, which the next request never repeats, so its entry is never read. The block before it is
    # the next request's prefix: marked there, call N+1 reads what call N wrote.
    req = ResponseRequest('i', (Text('a'), Text('b'), Text('c')), SCHEMA)
    content = anthropic_messages.WIRE.respond(req, 'claude-x', 'https://api.anthropic.com/v1', 'k').body['messages'][0][
        'content'
    ]
    assert [b.get('cache_control') for b in content] == [None, {'type': 'ephemeral'}, None]
    one = ResponseRequest('i', (Text('a'),), SCHEMA)
    only = anthropic_messages.WIRE.respond(one, 'claude-x', 'https://api.anthropic.com/v1', 'k').body['messages'][0]
    assert 'cache_control' not in only['content'][0]


def test_anthropic_messages_binds_the_schema_as_structured_output_and_caches_the_prefix():
    # Claude 5.5 refuses a forced tool_choice ("type tool and any are not supported", measured
    # 2026-10-16); output_config.format is the documented schema binding on every current model.
    call = anthropic_messages.WIRE.respond(REQ, 'claude-x', 'https://api.anthropic.com/v1', 'k')
    assert call.headers['x-api-key'] == 'k'
    assert call.body['output_config']['format'] == {'type': 'json_schema', 'schema': SCHEMA}
    assert 'tools' not in call.body and 'tool_choice' not in call.body
    assert call.body['cache_control'] == {'type': 'ephemeral'}
    assert call.body['messages'][0]['content'][1]['source']['data'] == B64
    raw = {'content': [{'type': 'text', 'text': '{"ok": true}'}], 'stop_reason': 'end_turn'}
    assert anthropic_messages.WIRE.parse_response(raw) == {'ok': True}
    with pytest.raises(ProviderError, match='max_tokens'):
        anthropic_messages.WIRE.parse_response({'content': [], 'stop_reason': 'max_tokens'})
    with pytest.raises(ProviderError, match='JSON'):
        anthropic_messages.WIRE.parse_response({'content': [{'type': 'text', 'text': 'nope'}]})


def test_gemini():
    call = gemini.WIRE.respond(REQ, 'gemini-x', 'https://g/v1beta', 'k')
    assert call.url == 'https://g/v1beta/models/gemini-x:generateContent'
    assert call.body['generationConfig']['responseJsonSchema'] == SCHEMA
    assert call.body['contents'][0]['parts'][1] == {'inlineData': {'mimeType': 'image/png', 'data': B64}}
    raw = {'candidates': [{'content': {'parts': [{'text': '{"ok":'}, {'text': ' true}'}]}}]}
    assert gemini.WIRE.parse_response(raw) == {'ok': True}
    with pytest.raises(ProviderError, match='SAFETY'):
        gemini.WIRE.parse_response({'candidates': [{'finishReason': 'SAFETY'}]})


def test_openai_chat_json_asks_for_a_json_object_and_carries_the_schema_in_the_prompt():
    call = openai_chat_json.WIRE.respond(REQ, 'deepseek-v4-flash', 'https://api.deepseek.com/v1', 'k')
    assert call.body['response_format'] == {'type': 'json_object'}
    system = call.body['messages'][0]['content']
    assert system.startswith('Be brief.')
    assert json.dumps(SCHEMA, sort_keys=True) in system
    assert openai_chat_json.WIRE.parse_response({'choices': [{'message': {'content': '{"ok": true}'}}]}) == {'ok': True}


def test_openai_chat_json_effort_off_turns_thinking_off_and_a_level_sets_the_effort():
    call = openai_chat_json.WIRE.respond(REQ, 'deepseek-v4-flash', 'https://api.deepseek.com/v1', 'k')
    off = openai_chat_json.WIRE.effort(call, 'off').body
    assert off['thinking'] == {'type': 'disabled'}
    assert 'reasoning_effort' not in off
    high = openai_chat_json.WIRE.effort(call, 'high').body
    assert high['thinking'] == {'type': 'enabled'}
    assert high['reasoning_effort'] == 'high'
