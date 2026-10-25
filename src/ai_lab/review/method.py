"""The sharp-review method: the packaged copy by default, any directory given in place; unknown versions refused.

The method is ai-lab's data (PLAN-a-short-robust-toolchain P4): every executor -- this package's headless run, the
ONE merge, sharp-review's Node planner -- reads :data:`METHOD_DIR`. ``python -m ai_lab.review.method`` prints it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = ['METHOD_DIR', 'METHOD_SCHEMA_VERSIONS', 'Method', 'UnknownMethodVersion', 'load_method', 'main']

#: The packaged method directory (package data, shipped in the wheel).
METHOD_DIR: Final = Path(__file__).with_name('method_data')

#: The ``method.json`` versions this executor reads; any other is refused, never guessed.
METHOD_SCHEMA_VERSIONS: Final = frozenset({2})


class UnknownMethodVersion(ValueError):
    """The method directory declares a ``schema_version`` this executor does not read."""


@dataclass(frozen=True)
class Method:
    """The parsed method: ``spec`` is ``method.json``; the rest are the files it names."""

    root: Path
    spec: dict
    profiles: tuple[dict, ...]
    finding_schema: dict
    #: The finding schema a reviewer prompt shows: ``method.json principles.prompt_schema`` picks filled or unfilled.
    prompt_schema: dict
    output_schema: dict
    #: ``principles.json``'s entries: the principle set every reviewer judges against.
    principles: tuple[dict, ...]

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


def load_method(root: str | Path = METHOD_DIR) -> Method:
    """The method at *root*; :class:`UnknownMethodVersion` naming the version, ``FileNotFoundError`` naming the file."""
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
    raw_schema = _json(root / files['finding_schema'])
    finding_schema = _with_principle_ids(raw_schema, spec['principles'], principles)
    prompt_schemas = {'filled': finding_schema, 'unfilled': raw_schema}
    choice = spec['principles'].get('prompt_schema')
    if choice not in prompt_schemas:
        msg = f'{spec_file} principles.prompt_schema is {choice!r}; it must be one of {sorted(prompt_schemas)}'
        raise ValueError(msg)
    return Method(
        root=root,
        spec=spec,
        profiles=tuple(_json(root / files['profiles'])['profiles']),
        finding_schema=finding_schema,
        prompt_schema=prompt_schemas[choice],
        output_schema=_json(root / files['output_schema']),
        principles=principles,
    )


def _with_principle_ids(schema: dict, spec: dict, principles: tuple[dict, ...]) -> dict:
    """``method.json principles.fills``: the schema names the field, ``principles.json`` alone lists its ids."""
    field = spec['finding_field']
    if field not in schema['properties']:
        msg = f'the finding schema has no {field!r} property for principles.json to fill'
        raise ValueError(msg)
    ids = [p['id'] for p in principles]
    if len(set(ids)) != len(ids):
        msg = f'principles.json repeats an id: {ids}'
        raise ValueError(msg)
    rule = schema['properties'][field]
    filled = {**rule, 'items': {**rule.get('items', {}), 'enum': ids}}
    return {**schema, 'properties': {**schema['properties'], field: filled}}


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
