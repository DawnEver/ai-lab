"""OpenAI ``POST /v1/decisions``: native predicate / choice / score with probabilities."""

from __future__ import annotations

from ai_lab.answers import YES_AT, Answer, ChoiceAnswer, PredicateAnswer, Refusal, ScoreAnswer
from ai_lab.errors import ProviderError
from ai_lab.spec import Choice, DecisionRequest, Image, Predicate, Question
from ai_lab.transport import HttpCall
from ai_lab.wire.base import Wire, text_of

__all__ = ['WIRE']


def _question(q: Question) -> dict:
    if isinstance(q, Predicate):
        return {'type': 'predicate', 'name': q.name, 'instructions': q.instructions}
    if isinstance(q, Choice):
        return {
            'type': 'choice',
            'name': q.name,
            'instructions': q.instructions,
            'choices': [{'value': o.value, 'description': o.description} for o in q.options],
        }
    return {
        'type': 'score',
        'name': q.name,
        'instructions': q.instructions,
        'levels': [{'label': lv.label, 'description': lv.description} for lv in q.levels],
    }


def _input(request: DecisionRequest) -> str | list:
    if not request.has_image:
        return '\n\n'.join(text_of(p) for p in request.context)
    content = [
        {'type': 'input_image', 'image_url': p.data_url()}
        if isinstance(p, Image)
        else {'type': 'input_text', 'text': text_of(p)}
        for p in request.context
    ]
    return [{'role': 'user', 'content': content}]


def _decide(request: DecisionRequest, model: str, endpoint: str, key: str) -> HttpCall:
    return HttpCall(
        url=f'{endpoint}/decisions',
        body={'model': model, 'input': _input(request), 'questions': [_question(q) for q in request.questions]},
        headers={'Authorization': f'Bearer {key}'},
    )


def _parse_decision(raw: dict, request: DecisionRequest) -> dict[str, Answer]:
    by_name = {a.get('name'): a for a in raw.get('answers', [])}
    answers: dict[str, Answer] = {}
    for q in request.questions:
        a = by_name.get(q.name)
        if a is None:
            msg = f'the decision has no answer for {q.name!r}: {raw}'
            raise ProviderError(msg)
        if a['type'] == 'refusal':
            answers[q.name] = Refusal(q.name)
        elif a['type'] == 'predicate':
            p = float(a['probability'])
            answers[q.name] = PredicateAnswer(q.name, p >= YES_AT, p)
        elif a['type'] == 'choice':
            probs = {str(e['value']): float(e['probability']) for e in a.get('probabilities', [])}
            answers[q.name] = ChoiceAnswer(q.name, a['choice'], probs or None, a.get('confidence'))
        elif a['type'] == 'score':
            probs = {e['label']: float(e['probability']) for e in a.get('probabilities', [])}
            labels = [lv.label for lv in q.levels]
            level = max(probs, key=probs.get) if probs else labels[round(float(a['score']))]
            answers[q.name] = ScoreAnswer(q.name, level, float(a['score']), probs or None, a.get('confidence'))
        else:
            msg = f'unknown answer type {a["type"]!r} for {q.name!r}'
            raise ProviderError(msg)
    return answers


WIRE = Wire(name='openai_decisions', cli=False, decide=_decide, parse_decision=_parse_decision)
