"""``ai_lab.review``: the sharp-review method and its headless executor.

The METHOD is pure data owned here (``method_data/``, PLAN-a-short-robust-toolchain P4): profiles, prompt
templates, the finding schema and the merge rules, under one ``schema_version``. It runs as a use of
``respond`` and knows nothing of forges. The rules the data cannot state are implemented against the
method's own ``conformance/`` fixtures. Its merge is the only one: sharp-review's Node executor reads the
method from ``python -m ai_lab.review.method`` and calls ``python -m ai_lab.review.merge`` rather than
keeping either.
"""

__all__ = []
