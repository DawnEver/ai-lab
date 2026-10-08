"""Every schema-responding wire: the schema reaches the vendor, images ride in its format, the
reply parses back to the object, and a refusal or malformed reply is a ProviderError."""

import base64
import json

import pytest

from ai_lab import Image, ProviderError, ResponseRequest, Text
from ai_lab.transport import CliResult
from ai_lab.wire import anthropic_messages, claude_code, codex, gemini, openai_chat, openai_responses
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


def test_anthropic_messages_forces_one_tool():
    call = anthropic_messages.WIRE.respond(REQ, 'claude-x', 'https://api.anthropic.com/v1', 'k')
    assert call.headers['x-api-key'] == 'k'
    assert call.body['tools'][0]['input_schema'] == SCHEMA
    assert call.body['tool_choice'] == {'type': 'tool', 'name': 'probe'}
    assert call.body['messages'][0]['content'][1]['source']['data'] == B64
    raw = {'content': [{'type': 'text', 'text': 'x'}, {'type': 'tool_use', 'name': 'probe', 'input': {'ok': True}}]}
    assert anthropic_messages.WIRE.parse_response(raw) == {'ok': True}
    with pytest.raises(ProviderError, match='tool_use'):
        anthropic_messages.WIRE.parse_response({'content': [], 'stop_reason': 'max_tokens'})


def test_gemini():
    call = gemini.WIRE.respond(REQ, 'gemini-x', 'https://g/v1beta', 'k')
    assert call.url == 'https://g/v1beta/models/gemini-x:generateContent'
    assert call.body['generationConfig']['responseJsonSchema'] == SCHEMA
    assert call.body['contents'][0]['parts'][1] == {'inlineData': {'mimeType': 'image/png', 'data': B64}}
    raw = {'candidates': [{'content': {'parts': [{'text': '{"ok":'}, {'text': ' true}'}]}}]}
    assert gemini.WIRE.parse_response(raw) == {'ok': True}
    with pytest.raises(ProviderError, match='SAFETY'):
        gemini.WIRE.parse_response({'candidates': [{'finishReason': 'SAFETY'}]})


def test_claude_code_runs_with_every_tool_and_setting_off():
    call = claude_code.WIRE.respond(REQ, 'haiku', 'claude', '')
    argv = list(call.argv)
    assert argv[0] == 'claude'
    assert argv[argv.index('--tools') + 1] == ''
    assert argv[argv.index('--setting-sources') + 1] == ''
    assert json.loads(argv[argv.index('--json-schema') + 1]) == SCHEMA
    assert argv[argv.index('--model') + 1] == 'haiku'
    message = json.loads(call.stdin)
    assert message['message']['content'][1]['source']['data'] == B64
    out = '{"type":"system"}\n{"type":"result","is_error":false,"structured_output":{"ok":true}}\n'
    assert claude_code.WIRE.parse_response(CliResult(out)) == {'ok': True}
    with pytest.raises(ProviderError, match='no structured output'):
        claude_code.WIRE.parse_response(CliResult('{"type":"result","is_error":true,"result":"boom"}'))


def test_claude_code_without_a_model_uses_the_cli_default():
    assert '--model' not in claude_code.WIRE.respond(REQ, '', 'claude', '').argv


def test_codex_is_read_only_and_reads_images_from_files():
    call = codex.WIRE.respond(REQ, '', 'codex', '')
    argv = list(call.argv)
    assert argv[:2] == ['codex', 'exec']
    assert argv[argv.index('--sandbox') + 1] == 'read-only'
    assert argv[argv.index('-i') + 1] == '{dir}/image1.png'
    assert call.files['image1.png'] == PNG
    assert json.loads(call.files['schema.json']) == SCHEMA
    assert call.stdin.startswith('Be brief.') and 'hello' in call.stdin
    assert argv[-1] == '-'
    assert codex.WIRE.parse_response(CliResult('{"ok": true}')) == {'ok': True}
