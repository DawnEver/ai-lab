"""ai-lab's runtime is the standard library, lab-commons and platformdirs -- nothing else.

Imported in a fresh interpreter whose finder refuses every other top-level name, so an array library
or a vendor SDK arriving anywhere in the import chain reds here rather than in a consumer's install.
"""

import subprocess
import sys

_ALLOWED = ('ai_lab', 'lab_commons', 'platformdirs')

_PROBE = """
import importlib.abc, sys
OK = set(sys.stdlib_module_names) | set(sys.argv[1].split(','))
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] not in OK:
            raise ModuleNotFoundError(f'{name!r} is outside the runtime closure')
sys.meta_path.insert(0, Refuse())
import ai_lab, ai_lab.client, ai_lab.transport
"""


def _probe(allowed: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', _PROBE, ','.join(allowed)],
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=False,
        timeout=120,
    )


def test_ai_lab_imports_with_only_its_declared_runtime():
    result = _probe(_ALLOWED)
    assert result.returncode == 0, result.stderr


def test_the_probe_refuses_a_missing_dependency():
    """THE PLANTED CONTROL: without platformdirs the same import must fail."""
    assert _probe(('ai_lab', 'lab_commons')).returncode != 0
