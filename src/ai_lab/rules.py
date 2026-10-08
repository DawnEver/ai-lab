"""A decider that is code, not a model: the offline baseline every model decider is measured against.

A rule reads the merged :class:`~ai_lab.spec.Fields` of the context and returns the answer's
VALUE -- a bool for a predicate, an option value for a choice, a level label for a score. A
question with no rule is refused, never guessed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ai_lab.answers import Answer, Basis, Decision
from ai_lab.emulate import parse_answers
from ai_lab.errors import Unsupported
from ai_lab.spec import DecisionRequest, Fields

__all__ = ['Rules']

Rule = Callable[[Mapping[str, object]], object]


class Rules:
    def __init__(self, rules: Mapping[str, Rule], *, name: str = 'rules') -> None:
        self.rules = dict(rules)
        self.name = name

    def decide(self, request: DecisionRequest) -> Decision:
        missing = [q.name for q in request.questions if q.name not in self.rules]
        if missing:
            raise Unsupported(f'{self.name} has no rule for {missing}; declare one per question')
        fields: dict[str, object] = {}
        for part in request.context:
            if isinstance(part, Fields):
                fields.update(part.values)
        values = {q.name: self.rules[q.name](fields) for q in request.questions}
        answers: dict[str, Answer] = parse_answers(values, request.questions)
        return Decision(answers, Basis.RULE, self.name, '')
