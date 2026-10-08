"""Real calls. Deselected by default; ``pytest -m live``. A provider whose CLI or key is absent
FAILS here with the Unsupported message naming what is missing -- it is never skipped."""

import pytest

from ai_lab import Basis, ResponseRequest, Text, connect

pytestmark = pytest.mark.live

SCHEMA = {
    'type': 'object',
    'properties': {'prime': {'type': 'boolean'}},
    'required': ['prime'],
    'additionalProperties': False,
}


@pytest.mark.parametrize('spec', ['claude_code:haiku', 'codex'])
def test_local_cli_responds_and_decides(spec, request_):
    client = connect(spec)
    assert client.respond(ResponseRequest('Answer.', (Text('Is 9 prime?'),), SCHEMA)).value == {'prime': False}
    d = client.decide(request_)
    assert d.basis is Basis.STATED and set(d.answers) == {'worth_fea', 'fidelity', 'risk'}


@pytest.mark.parametrize('spec', ['claude_code:haiku', 'codex'])
def test_local_cli_reads_an_image(spec):
    from tests.live.images import red_square

    schema = {
        'type': 'object',
        'properties': {'colour': {'type': 'string', 'enum': ['red', 'green', 'blue']}},
        'required': ['colour'],
        'additionalProperties': False,
    }
    got = connect(spec).respond(
        ResponseRequest('Name the colour.', (Text('What colour fills this image?'), red_square()), schema)
    )
    assert got.value == {'colour': 'red'}


def test_openai_decisions_native(request_):
    d = connect('openai_decisions:gpt-6-luna').decide(request_)
    assert d.basis is Basis.NATIVE
    assert d['fidelity'].probabilities is not None
