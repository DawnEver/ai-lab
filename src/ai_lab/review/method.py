"""One sharp-review method directory, loaded from the path given and refused when its version is unknown."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = ['METHOD_SCHEMA_VERSIONS', 'Method', 'UnknownMethodVersion', 'load_method']

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


def load_method(root: str | Path) -> Method:
    """The method at *root*; :class:`UnknownMethodVersion` naming the version, ``FileNotFoundError`` naming the file."""
    root = Path(root)
    spec_file = root / 'method.json'
    if not spec_file.is_file():
        msg = f'{spec_file} not found: --method must name a sharp-review method directory'
        raise FileNotFoundError(msg)
    spec = _json(spec_file)
    version = spec.get('schema_version')
    if version not in METHOD_SCHEMA_VERSIONS:
        msg = f'{spec_file} declares schema_version {version!r}; this executor reads {sorted(METHOD_SCHEMA_VERSIONS)}'
        raise UnknownMethodVersion(msg)
    files = spec['files']
    principles = tuple(_json(root / files['principles'])['principles'])
    return Method(
        root=root,
        spec=spec,
        profiles=tuple(_json(root / files['profiles'])['profiles']),
        finding_schema=_with_principle_ids(_json(root / files['finding_schema']), spec['principles'], principles),
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
