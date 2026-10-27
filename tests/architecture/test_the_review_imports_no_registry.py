"""ai-lab is the review METHOD; dev-lab RUNS it. The method imports nothing of its runner: the rule ids are given.

dev-lab's review command passes the registry ids (``--rules``, its ``{rules}`` placeholder); an ``import dev_lab``
(or an ``importlib`` string naming it) anywhere under ``src/ai_lab`` inverts that layering and is refused here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

_SRC: Final = Path(__file__).resolve().parents[2] / 'src' / 'ai_lab'

#: The shipped modules read; fewer means the walk missed the package.
_FLOOR: Final = 18

#: The runner's package: no import of it, and no string naming it as a module, ships in ai-lab.
_RUNNER: Final = re.compile(r'^dev_lab(\.|$)')


def runner_imports(source: str) -> list[str]:
    """Every module *source* imports (or names in a string as a dotted module) that is the runner's."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.append(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and ' ' not in node.value:
            found.append(node.value)
    return [name for name in found if _RUNNER.match(name)]


def test_the_scan_sees_an_import_of_the_runner() -> None:
    """PLANTED CONTROL: each spelling of the inversion is caught; a mention in prose is not an import."""
    assert runner_imports('import dev_lab.dev.rules') == ['dev_lab.dev.rules']
    assert runner_imports('from dev_lab.dev.rules import RULES') == ['dev_lab.dev.rules']
    assert runner_imports("importlib.import_module('dev_lab.dev.rules')") == ['dev_lab.dev.rules']
    assert runner_imports('"""dev-lab runs the review; dev_lab fills {rules} for it."""') == []


def test_no_shipped_module_imports_the_runner() -> None:
    modules = sorted(_SRC.rglob('*.py'))
    assert len(modules) >= _FLOOR, f'read only {len(modules)} modules under {_SRC}'
    found = {str(p.relative_to(_SRC)): hits for p in modules if (hits := runner_imports(p.read_text(encoding='utf-8')))}
    assert not found, f'ai-lab imports its runner (pass the value from the caller instead): {found}'
