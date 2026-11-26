"""ai-lab's ENV-IS-READ-ONCE entry (``[tool.dev_lab.env] entry``): the one module that reads the environment.

The pattern is :mod:`lab_commons.hostenv`'s: the facts ai-lab reads by NAME become the typed :class:`Env` record at
one door, :func:`context`, and a caller reads the record instead of the environment. Lab-commons' own host facts stay
in :mod:`lab_commons.hostenv`; these are the ones only ai-lab names:

* :data:`HOME_ENV` -- relocates ai-lab's per-user state (a test's own home, which its children inherit);
* :data:`TAG_ENV` -- what a call is charged to, set by the harness that launches a review;
* :data:`SCOPE_ENV` -- the allotment a call is charged in, the cross-process contract a child inherits, so it is
  also WRITTEN here (:func:`set_scope`);
* the user name (``USERNAME``, else ``USER``) an approval records by default.

:func:`environment` is the copy a launcher hands its child.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ['HOME_ENV', 'SCOPE_ENV', 'TAG_ENV', 'Env', 'context', 'environment', 'set_scope']

#: Relocates ai-lab's per-user state directory.
HOME_ENV: Final = 'AI_LAB_HOME'
#: The environment variable naming what a call is charged to (a PR review sets ``<owner>/<repo>#<number>``).
TAG_ENV: Final = 'AI_LAB_SPEND_TAG'
#: The environment variable naming the scope a call is charged in: the cross-process contract.
SCOPE_ENV: Final = 'AI_LAB_SCOPE'


@dataclass(frozen=True, slots=True)
class Env:
    """The facts ai-lab reads by name; ``''`` where the environment does not say."""

    home: str = ''
    spend_tag: str = ''
    scope: str = ''
    user: str = ''


def context(env: Mapping[str, str] | None = None) -> Env:
    """The :class:`Env` record of *env* (default: this process's environment, read now)."""
    env = os.environ if env is None else env
    return Env(
        home=env.get(HOME_ENV, ''),
        spend_tag=env.get(TAG_ENV, ''),
        scope=env.get(SCOPE_ENV, ''),
        user=env.get('USERNAME') or env.get('USER') or '',
    )


def set_scope(name: str | None) -> None:
    """Charge this process and every child it starts to the scope *name*; ``None`` clears it."""
    if name is None:
        os.environ.pop(SCOPE_ENV, None)
    else:
        os.environ[SCOPE_ENV] = name


def environment() -> dict[str, str]:
    """A copy of this process's environment, for a child to inherit."""
    return dict(os.environ)
