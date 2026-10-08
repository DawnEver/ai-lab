import pytest

from ai_lab import Choice, DecisionRequest, Fields, Image, Level, Option, Predicate, Score, Text

PNG = b'\x89PNG\r\n\x1a\nfake'


@pytest.fixture
def questions():
    return (
        Predicate('worth_fea', 'Is this design worth an expensive FEA?'),
        Choice('fidelity', 'Which fidelity next?', (Option('drop', 'discard'), Option('fea2d'), Option('fea3d'))),
        Score('risk', 'Thermal risk', (Level('low'), Level('medium'), Level('high', 'magnet near demag'))),
    )


@pytest.fixture
def request_(questions):
    return DecisionRequest((Text('IPM rotor, 48 slots.'), Fields({'torque_nm': 327.0, 'ripple_pct': 4.8})), questions)


@pytest.fixture
def image_request(questions):
    return DecisionRequest((Text('flux plot'), Image(PNG)), questions)
