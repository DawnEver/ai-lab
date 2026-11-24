"""The ``noul`` question family -- Jev's shape, which Google's Decisions API also speaks.

Questions are a map ``{name: {type, instructions[, criteria]}}``: ``noul`` (yes/no), ``choice`` (criteria map
option -> description) and ``score`` (criteria list, lowest level first; a level's description travels inside its
criterion string). Answers come back under the same names; a score's level index maps back to the declared label.
"""

from __future__ import annotations

from ai_lab.answers import YES_AT, Answer, ChoiceAnswer, PredicateAnswer, Refusal, ScoreAnswer
from ai_lab.errors import ProviderError
from ai_lab.spec import Choice, DecisionRequest, Predicate, Question

__all__ = ['parse_decision', 'questions']


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


def questions(request: DecisionRequest) -> dict[str, dict]:
    """The request's questions as the family's ``{name: question}`` map."""
    return {q.name: _question(q) for q in request.questions}


def parse_decision(raw: dict, request: DecisionRequest) -> dict[str, Answer]:
    """Every question's answer off ``raw['answers']``; a missing or unknown one is a :class:`ProviderError`."""
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
