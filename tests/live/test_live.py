"""Real calls: the ``live`` part, produced only where the credential resource exists. A provider whose key is absent
FAILS here with the Unsupported message naming what is missing -- it is never skipped."""

import pytest

from ai_lab import Basis, ResponseRequest, Text, connect
from tests.live.images import red_square

pytestmark = pytest.mark.live

SCHEMA = {
    'type': 'object',
    'properties': {'prime': {'type': 'boolean'}},
    'required': ['prime'],
    'additionalProperties': False,
}


#: The API rows that respond in a schema and read images, one model each.
_API = ['anthropic:claude-sonnet-5-5']


@pytest.mark.parametrize('spec', _API)
def test_an_api_row_responds_and_decides(spec, request_):
    client = connect(spec)
    assert client.respond(ResponseRequest('Answer.', (Text('Is 9 prime?'),), SCHEMA)).value == {'prime': False}
    d = client.decide(request_)
    assert d.basis is Basis.STATED and set(d.answers) == {'worth_fea', 'fidelity', 'risk'}


@pytest.mark.parametrize('spec', _API)
def test_an_api_row_reads_an_image(spec):
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
