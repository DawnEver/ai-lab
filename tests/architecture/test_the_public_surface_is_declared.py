"""Every shipped module declares ``__all__``, and each public name has exactly one home."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Final

_SRC: Final = Path(__file__).resolve().parents[2] / 'src' / 'ai_lab'

#: Names every module of one package declares by CONVENTION, read by that package's table rather than
#: imported by name: each ``ai_lab.wire`` protocol module is one ``WIRE`` record in ``WIRES``.
_PER_MODULE_RECORDS: Final = frozenset({'WIRE'})

#: The shipped modules; fewer means the walk missed the package.
_MODULE_FLOOR: Final = 18


def _declared(path: Path) -> list[str] | None:
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == '__all__' for t in targets):
            return list(ast.literal_eval(node.value))
    return None


def test_every_module_declares_its_surface_and_no_name_has_two_homes() -> None:
    modules = sorted(p for p in _SRC.rglob('*.py') if p.name != '__version__.py')
    assert len(modules) >= _MODULE_FLOOR, f'read only {len(modules)} modules under {_SRC}'
    undeclared = [str(p.relative_to(_SRC)) for p in modules if _declared(p) is None]
    assert not undeclared, f'modules without __all__: {undeclared}'
    homes: dict[str, list[str]] = defaultdict(list)
    for path in modules:
        if path.name == '__init__.py':
            continue
        for name in set(_declared(path) or []) - _PER_MODULE_RECORDS:
            homes[name].append(str(path.relative_to(_SRC)))
    twice = {name: where for name, where in homes.items() if len(where) > 1}
    assert not twice, f'one public name, several homes: {twice}'
