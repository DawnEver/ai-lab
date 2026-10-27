"""The sharp-review method: the packaged copy by default, any directory given in place; unknown versions refused.

The method is ai-lab's data (PLAN-a-short-robust-toolchain P4): every executor -- this package's headless run, the
ONE merge, sharp-review's Node planner -- reads :data:`METHOD_DIR`. ``python -m ai_lab.review.method`` prints it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    'METHOD_DIR',
    'METHOD_SCHEMA_VERSIONS',
    'NO_RULES',
    'Method',
    'UnknownMethodVersion',
    'load_method',
    'rule_ids',
]

#: The packaged method directory (package data, shipped in the wheel).
METHOD_DIR: Final = Path(__file__).with_name('method_data')

#: The ``method.json`` versions this executor reads; any other is refused, never guessed.
METHOD_SCHEMA_VERSIONS: Final = frozenset({3})


class UnknownMethodVersion(ValueError):
    """The method directory declares a ``schema_version`` this executor does not read."""


@dataclass(frozen=True)
class Method:
    """The parsed method: ``spec`` is ``method.json``; the rest are the files it names."""

    root: Path
    spec: dict
    profiles: tuple[dict, ...]
    finding_schema: dict
    #: The finding schema a reviewer prompt shows: ``method.json rules.prompt_schema`` picks filled or unfilled.
    prompt_schema: dict
    output_schema: dict
    #: ``principles.json``'s entries: the principles every reviewer judges against.
    principles: tuple[dict, ...]
    #: The rule-id set the CALLER gave (the family registry's, passed in): the rest of the valid ``rule`` set.
    registry: tuple[str, ...]

    @property
    def schema_version(self) -> int:
        """The method's version, which is also the findings JSON's."""
        return self.spec['schema_version']

    def text(self, relative: str) -> str:
        """A file of the method (a template, a partial) as text."""
        return (self.root / relative).read_text(encoding='utf-8')

    def profile(self, key: str) -> dict:
        """The profile named *key*; ``KeyError`` naming the known keys."""
        for row in self.profiles:
            if row['key'] == key:
                return row
        msg = f'no profile {key!r}; the method declares {[p["key"] for p in self.profiles]}'
        raise KeyError(msg)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


#: What an invocation without ``--rules`` is refused with: the ids come from the caller, never from an import.
NO_RULES: Final = (
    'no rule ids given: the caller passes the registry ids as --rules ID,ID,... '
    "(dev-lab's review command fills {rules}); ai-lab never imports the registry"
)


def rule_ids(given: str | None) -> tuple[str, ...]:
    """``--rules`` as ids (comma-separated; ``''`` is none); ``ValueError`` (:data:`NO_RULES`) when not given."""
    if given is None:
        raise ValueError(NO_RULES)
    return tuple(one.strip() for one in given.split(',') if one.strip())


def load_method(root: str | Path = METHOD_DIR, *, registry: Iterable[str]) -> Method:
    """The method at *root*; :class:`UnknownMethodVersion` naming the version, ``FileNotFoundError`` naming the file.

    *registry* is the rule-id set beside the principles, given by the caller: ai-lab imports no registry.
    """
    root = Path(root)
    spec_file = root / 'method.json'
    if not spec_file.is_file():
        msg = (
            f'{spec_file} not found: --method must name a sharp-review method directory (omit it for the packaged one)'
        )
        raise FileNotFoundError(msg)
    spec = _json(spec_file)
    version = spec.get('schema_version')
    if version not in METHOD_SCHEMA_VERSIONS:
        msg = f'{spec_file} declares schema_version {version!r}; this executor reads {sorted(METHOD_SCHEMA_VERSIONS)}'
        raise UnknownMethodVersion(msg)
    files = spec['files']
    principles = tuple(_json(root / files['principles'])['principles'])
    ids = tuple(registry)
    raw_schema = _json(root / files['finding_schema'])
    finding_schema = _with_rule_ids(raw_schema, spec['rules'], [p['id'] for p in principles] + list(ids))
    prompt_schemas = {'filled': finding_schema, 'unfilled': raw_schema}
    choice = spec['rules'].get('prompt_schema')
    if choice not in prompt_schemas:
        msg = f'{spec_file} rules.prompt_schema is {choice!r}; it must be one of {sorted(prompt_schemas)}'
        raise ValueError(msg)
    return Method(
        root=root,
        spec=spec,
        profiles=tuple(_json(root / files['profiles'])['profiles']),
        finding_schema=finding_schema,
        prompt_schema=prompt_schemas[choice],
        output_schema=_json(root / files['output_schema']),
        principles=principles,
        registry=ids,
    )


def _with_rule_ids(schema: dict, spec: dict, ids: Sequence[str]) -> dict:
    """``method.json rules.fills``: the schema names the field; the principles and the registry alone list its ids."""
    field = spec['finding_field']
    if field not in schema['properties']:
        msg = f'the finding schema has no {field!r} property for the rule ids to fill'
        raise ValueError(msg)
    if repeated := sorted({one for one in ids if ids.count(one) > 1}):
        msg = f'a rule id is declared twice (principles.json and the registry together): {repeated}'
        raise ValueError(msg)
    return {**schema, 'properties': {**schema['properties'], field: {**schema['properties'][field], 'enum': list(ids)}}}


def main(argv: Sequence[str] | None = None) -> int:
    """Print :data:`METHOD_DIR`, the directory a host executor reads the method from."""
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        sys.stderr.write(f'python -m ai_lab.review.method takes no arguments; got {list(argv)}\n')
        return 2
    sys.stdout.write(f'{METHOD_DIR}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
