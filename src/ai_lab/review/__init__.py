"""``ai_lab.review``: the headless executor of sharp-review's method.

The METHOD is pure data owned by sharp-review (cc-market ``sharp-review/method/``): profiles, prompt
templates, the finding schema and the merge rules, under one ``schema_version``. This package reads that
directory from the path it is given -- never a vendored copy -- and runs it as a use of ``respond``; it
knows nothing of forges. The rules the data cannot state are implemented against the method's own
``conformance/`` fixtures. Its merge is the only one: sharp-review's Node executor calls
``python -m ai_lab.review merge`` rather than keeping its own.
"""

__all__ = []
