"""The principles the review tests give as the caller does: ai-lab keeps none, so the tests supply dev-lab's set.

``fixtures/principles.json`` is the family's list as of 2026-10-27; the conformance fixtures render and cite it.
``fixtures/no-requirements.json`` is ``{}``: a caller declaring no requirement, which every entry point still requires
be PASSED (``--requirements``), so a test that is not about requirements says so explicitly.
"""

from pathlib import Path

from ai_lab.review.method import principle_list

PRINCIPLES_FILE = Path(__file__).with_name('fixtures') / 'principles.json'
PRINCIPLES = principle_list(PRINCIPLES_FILE)
REQUIREMENTS_FILE = Path(__file__).with_name('fixtures') / 'no-requirements.json'
#: What every review entry point is given by its caller: the principles, and no requirement.
CALLER_ARGS = ('--principles', str(PRINCIPLES_FILE), '--requirements', str(REQUIREMENTS_FILE))
