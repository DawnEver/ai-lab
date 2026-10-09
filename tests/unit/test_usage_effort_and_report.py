"""Usage is read per wire, effort is part of the client's name, and a ledger reports both."""

from __future__ import annotations

import json

import pytest

from ai_lab import DecisionRequest, Ledger, Predicate, ResponseRequest, Text, connect
from ai_lab.errors import Unsupported
from ai_lab.prices import Price, Quote
from ai_lab.report import summarize
from ai_lab.transport import CliCall, HttpCall
from ai_lab.usage import (
    Usage,
    anthropic_usage,
    chat_usage,
    claude_code_usage,
    gemini_usage,
    responses_usage,
)
from ai_lab.wire import WIRES
from ai_lab.wire.base import body_effort

REQUEST = DecisionRequest((Text('x0=0.2'),), (Predicate('p0', 'Is x0 below 0.5?'),))
DECISIONS_RAW = {
    'answers': [{'type': 'predicate', 'name': 'p0', 'value': True, 'probability': 0.9}],
    'usage': {
        'input_tokens': 163,
        'input_tokens_details': {'cached_tokens': 3},
        'output_tokens': 0,
        'output_tokens_details': {'reasoning_tokens': 0},
    },
}


def test_each_vendor_shape_is_read_into_one_usage() -> None:
    assert responses_usage(DECISIONS_RAW) == Usage(163, 0, 3, 0)
    chat = {
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'completion_tokens_details': {'reasoning_tokens': 2}}
    }
    assert chat_usage(chat) == Usage(10, 5, 0, 2)
    assert chat_usage({'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'prompt_cache_hit_tokens': 4}}) == Usage(
        10, 5, 4, 0
    )
    anthropic = {'usage': {'input_tokens': 7, 'cache_read_input_tokens': 3, 'output_tokens': 4}}
    assert anthropic_usage(anthropic) == Usage(10, 4, 3, 0)
    gemini = {'usageMetadata': {'promptTokenCount': 9, 'candidatesTokenCount': 2, 'thoughtsTokenCount': 6}}
    assert gemini_usage(gemini) == Usage(9, 8, 0, 6)
    assert claude_code_usage({**anthropic, 'total_cost_usd': 0.0123}).cost_usd == 0.0123
    assert responses_usage({'answers': []}) is None


def test_every_http_wire_reads_usage_and_every_effort_hook_is_declared_data() -> None:
    assert {n for n, w in WIRES.items() if w.parse_usage is None} == {'codex'}
    assert {n for n, w in WIRES.items() if w.effort} == {'openai_responses', 'openai_chat', 'openai_chat_json', 'codex'}


def test_a_native_decision_carries_the_vendors_usage(monkeypatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'k')
    decision = connect('openai_decisions:m', post=lambda call: DECISIONS_RAW).decide(REQUEST)
    assert decision.usage == Usage(163, 0, 3, 0)


def test_effort_is_part_of_the_name_and_reaches_the_body(monkeypatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'k')
    seen = []

    def post(call):
        seen.append(call)
        return {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"a": 1}'}]}]}

    client = connect('openai:gpt-x@high', post=post)
    assert client.name == 'openai:gpt-x@high'
    client.respond(ResponseRequest('r', (Text('t'),), {'type': 'object'}))
    assert seen[0].body['reasoning'] == {'effort': 'high'}


def test_a_wire_with_no_effort_refuses_one() -> None:
    with pytest.raises(Unsupported, match='takes no effort'):
        connect('openai_decisions:m@high')


def test_codex_effort_goes_before_the_prompt_dash() -> None:
    call = CliCall(argv=('codex', 'exec', '-'), stdin='')
    assert WIRES['codex'].effort(call, 'low').argv == ('codex', 'exec', '-c', 'model_reasoning_effort="low"', '-')


def test_a_body_effort_nests_without_touching_the_original() -> None:
    call = HttpCall(url='u', body={'model': 'm', 'reasoning': {'summary': 'auto'}})
    out = body_effort('reasoning', 'effort')(call, 'low')
    assert out.body['reasoning'] == {'summary': 'auto', 'effort': 'low'}
    assert call.body['reasoning'] == {'summary': 'auto'}


def test_the_ledger_records_usage_and_the_report_totals_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'k')
    ledger = Ledger(tmp_path / 'l.jsonl')
    recorded = ledger.wrap(connect('openai_decisions:m', post=lambda call: DECISIONS_RAW))
    recorded.decide(REQUEST)
    entry = json.loads((tmp_path / 'l.jsonl').read_text('utf-8'))
    assert entry['client'] == 'openai_decisions:m' and entry['usage']['input_tokens'] == 163
    replayed = Ledger(tmp_path / 'l.jsonl', 'replay').wrap(connect('openai_decisions:m')).decide(REQUEST)
    assert replayed.usage == Usage(163, 0, 3, 0)
    assert 'at' in entry
    flat = Quote(Price(1.0, 0.0), 'test', agreed=True, sources={})
    (row,) = summarize([tmp_path / 'l.jsonl'], price=lambda name, at: flat)
    assert (row.client, row.calls, row.questions, row.input_tokens, row.price) == (
        'openai_decisions:m',
        1,
        1,
        163,
        'test',
    )
    assert row.cost_usd == pytest.approx(160 / 1e6 + 3 / 1e6)
