"""The principles the review tests give as the caller does: ai-lab keeps none, so the tests supply dev-lab's set.

``fixtures/principles.json`` is the family's list as of 2026-10-27; the conformance fixtures render and cite it.
"""

from pathlib import Path

from ai_lab.review.method import principle_list

PRINCIPLES_FILE = Path(__file__).with_name('fixtures') / 'principles.json'
PRINCIPLES = principle_list(PRINCIPLES_FILE)
