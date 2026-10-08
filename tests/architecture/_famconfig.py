"""ai-lab's delta against each family config base -- data, computed by nothing.

The bases live in ``lab_commons.dev._famconfig_rows`` and :func:`lab_commons.dev.famconfig.render`
turns base plus delta into the file. ai-lab adds nothing to any base: everything it ignores, hooks and
verifies is the family's, so every delta is empty and a fix upstream lands here by re-rendering.
"""

from __future__ import annotations

from typing import Final

from lab_commons.dev.famconfig import Delta

__all__ = ['DELTAS', 'REPO']

REPO: Final = 'ai-lab'

DELTAS: Final[dict[str, Delta]] = {
    artefact: Delta(repo=REPO, added=(), dropped={}, ceiling=0)
    for artefact in ('.gitattributes', '.gitignore', '.pre-commit-config.yaml', '.rgignore', 'Makefile')
}
