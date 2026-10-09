"""TypeSafe ``POST /v1/systemone`` (Jev): native noul / choice / score with probabilities.

The state is sent as an OBJECT when the context is a single :class:`Fields`, else as text.
A score level's description travels inside its criterion string, which the API treats as the
level's meaning; the answer's level index maps back to the declared label.
"""

from __future__ import annotations

from ai_lab.answers import YES_AT, Answer, ChoiceAnswer, PredicateAnswer, Refusal, ScoreAnswer
from ai_lab.errors import ProviderError
from ai_lab.spec import Choice, DecisionRequest, Fields, Predicate, Question
from ai_lab.transport import HttpCall
from ai_lab.usage import responses_usage
from ai_lab.wire.base import Wire, joined_text

__all__ = ['WIRE']


def _question(q: Question) -> dict:
    if isinstance(q, Predicate):
        return {'type': 'noul', 'instructions': q.instructions}
    if isinstance(q, Choice):
        return {
            'type': 'choice',
            'instructions': q.instructions,
            'criteria': {o.value: o.description or o.value for o in q.options},
        }
    return {
        'type': 'score',
        'instructions': q.instructions,
        'criteria': [f'{lv.label}: {lv.description}' if lv.description else lv.label for lv in q.levels],
    }


def _decide(request: DecisionRequest, model: str, endpoint: str, key: str) -> HttpCall:
    context = request.context
    state = dict(context[0].values) if len(context) == 1 and isinstance(context[0], Fields) else joined_text(context)
    return HttpCall(
        url=f'{endpoint}/systemone',
        body={'model': model, 'state': state, 'questions': {q.name: _question(q) for q in request.questions}},
        headers={'Authorization': f'Bearer {key}'},
    )


def _parse_decision(raw: dict, request: DecisionRequest) -> dict[str, Answer]:
    got = raw.get('answers', {})
    answers: dict[str, Answer] = {}
    for q in request.questions:
        a = got.get(q.name)
        if a is None:
            msg = f'the decision has no answer for {q.name!r}: {raw}'
            raise ProviderError(msg)
        if a['type'] == 'refusal':
            answers[q.name] = Refusal(q.name)
        elif a['type'] == 'noul':
            p = float(a['noul'])
            answers[q.name] = PredicateAnswer(q.name, p >= YES_AT, p)
        elif a['type'] == 'choice':
            probs = {k: float(v) for k, v in a.get('probabilities', {}).items()}
            answers[q.name] = ChoiceAnswer(q.name, a['choice'], probs or None, a.get('confidence'))
        elif a['type'] == 'score':
            labels = [lv.label for lv in q.levels]
            probs = {labels[int(k)]: float(v) for k, v in a.get('probabilities', {}).items()}
            level = max(probs, key=probs.get) if probs else labels[round(float(a['score']))]
            answers[q.name] = ScoreAnswer(q.name, level, float(a['score']), probs or None, a.get('confidence'))
        else:
            msg = f'unknown answer type {a["type"]!r} for {q.name!r}'
            raise ProviderError(msg)
    return answers


#: Usage measured 2026-10-09: `usage.input_tokens` / `output_tokens`, the Responses shape.
WIRE = Wire(name='typesafe', cli=False, decide=_decide, parse_decision=_parse_decision, parse_usage=responses_usage)
