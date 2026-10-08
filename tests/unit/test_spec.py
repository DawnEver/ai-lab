import pytest

from ai_lab import Choice, DecisionRequest, Fields, Image, Option, Predicate, ResponseRequest, Text
from ai_lab.spec import canonical


def test_a_choice_needs_two_distinct_options():
    with pytest.raises(ValueError, match='two distinct'):
        Choice('c', 'pick', (Option('a'), Option('a')))


def test_question_names_are_distinct():
    p = Predicate('x', 'q')
    with pytest.raises(ValueError, match='distinct names'):
        DecisionRequest((Text('t'),), (p, p))


def test_a_context_part_has_a_declared_kind():
    with pytest.raises(TypeError, match='Text, Fields or Image'):
        DecisionRequest(('raw string',), (Predicate('x', 'q'),))


def test_an_image_needs_an_image_mime():
    with pytest.raises(ValueError, match='image/'):
        Image(b'x', 'text/plain')


def test_a_response_schema_is_an_object():
    with pytest.raises(ValueError, match='type: object'):
        ResponseRequest('i', (Text('t'),), {'type': 'string'})


def test_canonical_text_names_an_image_by_digest_not_bytes(image_request):
    text = canonical(image_request)
    assert 'sha256' in text
    assert 'fake' not in text


def test_canonical_text_is_order_free_for_fields():
    a = canonical(Fields({'a': 1, 'b': 2}))
    b = canonical(Fields({'b': 2, 'a': 1}))
    assert a == b
