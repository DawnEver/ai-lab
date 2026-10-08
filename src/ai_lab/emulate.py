"""A decision asked of a model that has no decision endpoint: one derived schema, one derived prompt.

Any wire that can respond in a JSON schema can therefore decide, and no wire writes this
translation twice. The answers are ``STATED``: the model named an option, it did not report a
distribution, so no probability is attached.
"""

from __future__ import annotations

from collections.abc import Mapping

from ai_lab.answers import Answer, ChoiceAnswer, PredicateAnswer, ScoreAnswer
from ai_lab.errors import ProviderError
from ai_lab.spec import Choice, DecisionRequest, Predicate, Question, ResponseRequest, Score

__all__ = ['as_response_request', 'decision_schema', 'parse_answers']

_INSTRUCTIONS = (
    'Answer every question about the context below. Return exactly the JSON object the schema '
    'declares: for a predicate true or false, for a choice one of its option values, for a score '
    'one of its level labels.'
)


def _describe(question: Question) -> str:
    if isinstance(question, Predicate):
        return f'{question.instructions} (true or false)'
    if isinstance(question, Choice):
        options = '; '.join(f'{o.value}: {o.description}' if o.description else o.value for o in question.options)
        return f'{question.instructions} Options -- {options}'
    levels = '; '.join(f'{lv.label}: {lv.description}' if lv.description else lv.label for lv in question.levels)
    return f'{question.instructions} Levels, lowest first -- {levels}'


def _property(question: Question) -> dict:
    if isinstance(question, Predicate):
        return {'type': 'boolean', 'description': _describe(question)}
    if isinstance(question, Choice):
        return {'type': 'string', 'enum': [o.value for o in question.options], 'description': _describe(question)}
    return {'type': 'string', 'enum': [lv.label for lv in question.levels], 'description': _describe(question)}


def decision_schema(questions: tuple[Question, ...]) -> dict:
    """A strict JSON schema (every property required, nothing additional) answering ``questions``."""
    return {
        'type': 'object',
        'properties': {q.name: _property(q) for q in questions},
        'required': [q.name for q in questions],
        'additionalProperties': False,
    }


def as_response_request(request: DecisionRequest) -> ResponseRequest:
    """The decision request as a response request carrying its derived schema."""
    return ResponseRequest(
        instructions=_INSTRUCTIONS,
        context=request.context,
        schema=decision_schema(request.questions),
        name='decision',
    )


def parse_answers(value: Mapping, questions: tuple[Question, ...]) -> dict[str, Answer]:
    """Typed answers from a schema-bound reply, refusing any value outside its question's set."""
    answers: dict[str, Answer] = {}
    for q in questions:
        if q.name not in value:
            msg = f'the response has no answer for question {q.name!r}: {dict(value)}'
            raise ProviderError(msg)
        got = value[q.name]
        if isinstance(q, Predicate):
            if not isinstance(got, bool):
                msg = f'predicate {q.name!r} needs true/false, got {got!r}'
                raise ProviderError(msg)
            answers[q.name] = PredicateAnswer(q.name, got)
        elif isinstance(q, Choice):
            if got not in {o.value for o in q.options}:
                msg = f'choice {q.name!r} got {got!r}, not one of its options'
                raise ProviderError(msg)
            answers[q.name] = ChoiceAnswer(q.name, got)
        elif isinstance(q, Score):
            labels = [lv.label for lv in q.levels]
            if got not in labels:
                msg = f'score {q.name!r} got {got!r}, not one of its levels {labels}'
                raise ProviderError(msg)
            answers[q.name] = ScoreAnswer(q.name, got, float(labels.index(got)))
    return answers
