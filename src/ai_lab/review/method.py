"""The sharp-review method: the packaged copy by default, any directory given in place; unknown versions refused.

The method is ai-lab's data (PLAN-a-short-robust-toolchain P4): every executor -- this package's headless run, the
ONE merge, sharp-review's Node planner -- reads :data:`METHOD_DIR`. ``python -m ai_lab.review.method`` prints it.

What a review is held to is the CALLER's, given at load and never kept here: the principles (``--principles``), the
registry ids (``--rules``) and the requirements (``--requirements``: every principle and/or each architecture
document, :func:`requirement_file`). Every entry point that loads the method refuses an invocation missing one.
"""

from __future__ import annotations

import json
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ai_lab.client import route

__all__ = [
    'ARCHITECTURE_PREFIX',
    'METHOD_DIR',
    'METHOD_SCHEMA_VERSIONS',
    'NO_PRINCIPLES',
    'NO_REQUIREMENTS',
    'NO_RULES',
    'REQUIREMENTS_HELP',
    'REQUIREMENT_KINDS',
    'REVIEW_TASK',
    'Method',
    'UnknownMethodVersion',
    'configured_providers',
    'default_config',
    'load_method',
    'principle_list',
    'requirement_file',
    'rule_ids',
]

#: The packaged method directory (package data, shipped in the wheel).
METHOD_DIR: Final = Path(__file__).with_name('method_data')

#: The task a reviewer's call runs as: its keys, and its route (``routes.toml``) from provider to provider.
REVIEW_TASK: Final = 'pr-review'


def default_config() -> dict:
    """The default configuration: every run mode reviews on the head of :data:`REVIEW_TASK`'s route.

    The route is the one place a review's provider is declared (user ruling 2026-11-05); ``providers`` stays a
    round-robin over reviewers, and each reviewer's call falls over along the same route. A project config
    (``.claude/sharp-review.json``) or an explicit request overrides it; the method keeps every mode.
    """
    head = route(REVIEW_TASK).chain[0][0]
    return {'providers': {'interactive': [head], 'headless': [head]}}


def configured_providers(config: Mapping, run_mode: str, defaults: Mapping | None = None) -> list[str] | None:
    """The providers *config* names for *run_mode*, else *defaults*' (:func:`default_config`); ``None`` when neither."""
    for source in (config, default_config() if defaults is None else defaults):
        named = (source.get('providers') or {}).get(run_mode)
        if named:
            return list(named)
    return None


#: The ``method.toml`` versions this executor reads; any other is refused, never guessed.
METHOD_SCHEMA_VERSIONS: Final = frozenset({3})


class UnknownMethodVersion(ValueError):
    """The method directory declares a ``schema_version`` this executor does not read."""


@dataclass(frozen=True)
class Method:
    """The parsed method: ``spec`` is ``method.toml``; the rest are the files it names."""

    root: Path
    spec: dict
    profiles: tuple[dict, ...]
    finding_schema: dict
    #: The finding schema a reviewer prompt shows: ``method.toml rules.prompt_schema`` picks filled or unfilled.
    prompt_schema: dict
    output_schema: dict
    #: The principles the CALLER gave (dev-lab owns the family's): every reviewer judges against them.
    principles: tuple[dict, ...]
    #: The rule-id set the CALLER gave (the family registry's, passed in): the rest of the valid ``rule`` set.
    registry: tuple[str, ...]
    #: The REQUIREMENTS the caller declared (``--requirements``): every reviewer returns a verdict on each. Rows
    #: ``{id, kind: principle, statement}`` or ``{id, kind: architecture, path, blob, text}``, in that order.
    requirements: tuple[dict, ...]

    @property
    def schema_version(self) -> int:
        """The method's version, which is also the findings JSON's."""
        return self.spec['schema_version']

    @property
    def rule_ids(self) -> tuple[str, ...]:
        """Every id a finding's ``rule`` may name: the principles, the registry, then the architecture documents."""
        documents = (row['id'] for row in self.requirements if row['kind'] == 'architecture')
        return (*(p['id'] for p in self.principles), *self.registry, *documents)

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


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding='utf-8'))


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


#: What an invocation without ``--principles`` is refused with: like the registry, the principles are the caller's.
NO_PRINCIPLES: Final = (
    '--principles FILE is required and was not passed: the caller (e.g. an older sharp-review plugin) must pass it, '
    'a JSON {"principles": [{id, name, statement}]} file '
    "(dev-lab owns the family's; an empty list reviews against none); ai-lab keeps no list of its own"
)

_PRINCIPLE_KEYS: Final = frozenset({'id', 'name', 'statement'})


def principle_list(given: str | Path | None) -> tuple[dict, ...]:
    """``--principles FILE`` as entries; ``ValueError`` (:data:`NO_PRINCIPLES`) when not given."""
    if given is None:
        raise ValueError(NO_PRINCIPLES)
    path = Path(given)
    if not path.is_file():
        msg = f'{path} not found: --principles must name a JSON file {{"principles": [...]}}'
        raise FileNotFoundError(msg)
    try:
        data = _json(path)
    except json.JSONDecodeError as bad:
        msg = f'{path} is not JSON: {bad}'
        raise ValueError(msg) from None
    if not isinstance(data, dict) or not isinstance(data.get('principles'), list):
        msg = f'{path} must be a JSON object with a "principles" list'
        raise TypeError(msg)
    return tuple(data['principles'])


#: What an invocation without ``--requirements`` is refused with: what a review is held to is the caller's.
NO_REQUIREMENTS: Final = (
    '--requirements FILE is required and was not passed: the caller passes a JSON object of the requirements it '
    'declares, {"principles": true, "architecture": [{id, path, blob, text}]} '
    '(dev-lab writes it per run from [tool.dev_lab.review] requirements; {} declares none); ai-lab keeps none'
)

#: The ``--requirements`` flag's help, the same on every entry point that loads the method.
REQUIREMENTS_HELP: Final = (
    'a JSON file {"principles": true, "architecture": [{id, path, blob, text}]}: what every reviewer returns a '
    'verdict on (required: the caller declares them; {} declares none)'
)

#: The requirement kinds a ``--requirements`` file may name, in the order their rows are listed.
REQUIREMENT_KINDS: Final = ('principles', 'architecture')

#: An architecture requirement's id: this prefix, then its document's path.
ARCHITECTURE_PREFIX: Final = 'arch:'

_DOCUMENT_KEYS: Final = frozenset({'id', 'path', 'blob', 'text'})


def requirement_file(given: str | Path | None) -> dict:
    """``--requirements FILE`` as the JSON object it holds; ``ValueError`` (:data:`NO_REQUIREMENTS`) when not given.

    Its rows are checked where the method reads them (:func:`load_method`), so an object handed in directly is held
    to the same shape.
    """
    if given is None:
        raise ValueError(NO_REQUIREMENTS)
    path = Path(given)
    if not path.is_file():
        msg = f'{path} not found: --requirements must name a JSON file {{"principles": true, "architecture": [...]}}'
        raise FileNotFoundError(msg)
    try:
        data = _json(path)
    except json.JSONDecodeError as bad:
        msg = f'{path} is not JSON: {bad}'
        raise ValueError(msg) from None
    if not isinstance(data, dict):
        msg = f'{path} must be a JSON object whose keys are among {list(REQUIREMENT_KINDS)}'
        raise TypeError(msg)
    return data


def _document(row: object) -> dict:
    """One architecture requirement, exactly ``{id, path, blob, text}``: strings, ``id`` the prefixed path, text."""
    if (
        not isinstance(row, dict)
        or set(row) != _DOCUMENT_KEYS
        or not all(isinstance(value, str) for value in row.values())
        or not row['path'].strip()
        or not row['text'].strip()
        or row['id'] != ARCHITECTURE_PREFIX + row['path']
    ):
        shown = {key: value for key, value in row.items() if key != 'text'} if isinstance(row, dict) else row
        msg = (
            f'an architecture requirement is exactly {sorted(_DOCUMENT_KEYS)}, all strings, id '
            f'"{ARCHITECTURE_PREFIX}<path>" and a non-empty text: {shown!r}'
        )
        raise ValueError(msg)
    return {**row, 'kind': 'architecture'}


def _requirements(given: Mapping, principles: Sequence[dict]) -> tuple[dict, ...]:
    """The rows *given* declares: every principle when ``principles`` is ``true``, then each architecture document."""
    if unknown := sorted(set(given) - set(REQUIREMENT_KINDS)):
        msg = f'--requirements names {unknown}: the requirement kinds are {list(REQUIREMENT_KINDS)}'
        raise ValueError(msg)
    if 'principles' in given and given['principles'] is not True:
        msg = f'--requirements principles is {given["principles"]!r}: true (every --principles entry) or absent'
        raise ValueError(msg)
    documents = given.get('architecture', [])
    if not isinstance(documents, list):
        msg = '--requirements architecture is not a list of {id, path, blob, text}'
        raise TypeError(msg)
    held = [{'id': p['id'], 'kind': 'principle', 'statement': p['statement']} for p in principles]
    rows = [*(held if given.get('principles') else ()), *map(_document, documents)]
    ids = [row['id'] for row in rows]
    if repeated := sorted({one for one in ids if ids.count(one) > 1}):
        msg = f'a requirement id is given twice: {repeated}'
        raise ValueError(msg)
    return tuple(rows)


def _checked(principles: Iterable[dict]) -> tuple[dict, ...]:
    """Every entry is exactly ``{id, name, statement}``, all non-empty strings; ids unique."""
    entries = tuple(principles)
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != _PRINCIPLE_KEYS
            or not all(isinstance(v, str) and v.strip() for v in entry.values())
        ):
            msg = f'a principle must be exactly {sorted(_PRINCIPLE_KEYS)}, each a non-empty string: {entry!r}'
            raise ValueError(msg)
    ids = [entry['id'] for entry in entries]
    if repeated := sorted({one for one in ids if ids.count(one) > 1}):
        msg = f'a principle id is given twice: {repeated}'
        raise ValueError(msg)
    return entries


def load_method(
    root: str | Path = METHOD_DIR, *, principles: Iterable[dict], registry: Iterable[str], requirements: Mapping
) -> Method:
    """The method at *root*; :class:`UnknownMethodVersion` naming the version, ``FileNotFoundError`` naming the file.

    *principles*, *registry* and *requirements* (the ``--requirements`` object, :func:`requirement_file`) are all the
    caller's (dev-lab owns them): ai-lab keeps no list and imports no registry. An architecture requirement's id joins
    the ids a finding's ``rule`` may name (:attr:`Method.rule_ids`).
    """
    root = Path(root)
    spec_file = root / 'method.toml'
    if not spec_file.is_file():
        msg = (
            f'{spec_file} not found: --method must name a sharp-review method directory (omit it for the packaged one)'
        )
        raise FileNotFoundError(msg)
    spec = _toml(spec_file)
    version = spec.get('schema_version')
    if version not in METHOD_SCHEMA_VERSIONS:
        msg = f'{spec_file} declares schema_version {version!r}; this executor reads {sorted(METHOD_SCHEMA_VERSIONS)}'
        raise UnknownMethodVersion(msg)
    files = spec['files']
    principles = _checked(principles)
    ids = tuple(registry)
    rows = _requirements(requirements, principles)
    documents = [row['id'] for row in rows if row['kind'] == 'architecture']
    raw_schema = _json(root / files['finding_schema'])
    finding_schema = _with_rule_ids(raw_schema, spec['rules'], [p['id'] for p in principles] + list(ids) + documents)
    prompt_schemas = {'filled': finding_schema, 'unfilled': raw_schema}
    choice = spec['rules'].get('prompt_schema')
    if choice not in prompt_schemas:
        msg = f'{spec_file} rules.prompt_schema is {choice!r}; it must be one of {sorted(prompt_schemas)}'
        raise ValueError(msg)
    return Method(
        root=root,
        spec=spec,
        profiles=tuple(_toml(root / files['profiles'])['profiles']),
        finding_schema=finding_schema,
        prompt_schema=prompt_schemas[choice],
        output_schema=_json(root / files['output_schema']),
        principles=principles,
        registry=ids,
        requirements=rows,
    )


def _with_rule_ids(schema: dict, spec: dict, ids: Sequence[str]) -> dict:
    """``method.toml rules.fills``: the schema names the field; the principles and the registry alone list its ids."""
    field = spec['finding_field']
    if field not in schema['properties']:
        msg = f'the finding schema has no {field!r} property for the rule ids to fill'
        raise ValueError(msg)
    if repeated := sorted({one for one in ids if ids.count(one) > 1}):
        msg = f'a rule id is declared twice (the principles, the registry and the documents together): {repeated}'
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
