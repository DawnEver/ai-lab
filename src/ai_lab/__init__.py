"""ai-lab: typed decisions and schema-bound responses from any model, through one interface.

Two verbs -- ``decide`` (closed questions -> typed answers, with a named probability basis) and
``respond`` (a context -> a JSON-schema object) -- over wires that are protocols, with vendors as
rows of data. See ``README.md``.
"""

from ai_lab.answers import (
    Answer,
    Basis,
    ChoiceAnswer,
    Decision,
    PredicateAnswer,
    Refusal,
    Response,
    ScoreAnswer,
)
from ai_lab.client import Client, Provider, connect, providers
from ai_lab.errors import ProviderError, ReplayMiss, Unsupported
from ai_lab.ledger import Ledger, Mode, Recorded
from ai_lab.protocols import Decider, Responder
from ai_lab.rules import Rules
from ai_lab.spec import (
    Choice,
    DecisionRequest,
    Fields,
    Image,
    Level,
    Option,
    Predicate,
    ResponseRequest,
    Score,
    Text,
)

__all__ = [
    'Answer',
    'Basis',
    'Choice',
    'ChoiceAnswer',
    'Client',
    'Decider',
    'Decision',
    'DecisionRequest',
    'Fields',
    'Image',
    'Ledger',
    'Level',
    'Mode',
    'Option',
    'Predicate',
    'PredicateAnswer',
    'Provider',
    'ProviderError',
    'Recorded',
    'Refusal',
    'ReplayMiss',
    'Responder',
    'Response',
    'ResponseRequest',
    'Rules',
    'Score',
    'ScoreAnswer',
    'Text',
    'Unsupported',
    'connect',
    'providers',
]
