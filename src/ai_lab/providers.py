"""``python -m ai_lab.providers --json`` -- the resolved provider catalogue, as one versioned document.

The catalogue is the packaged ``providers.toml`` with every row of the user's own (``user_providers_file``)
replacing its namesake (:func:`ai_lab.client.catalogue`). This prints that resolved table together with
what a consumer needs to know whether a copy of it is still current: the INPUT files and their
``sha256``, the ai-lab version, the interpreter that produced it and when. A host launcher's setup
cache is built from this output; it holds no secret -- a key is named by its credential USERNAME
(:func:`lab_commons.credentials.key_username`), never carried.

``source_sha256`` is the sha256 of the packaged file's bytes followed by the user file's bytes (none
when it is absent), so a consumer recomputes it from the two recorded paths without Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version

from lab_commons.credentials import API_KEY_TASKS, key_username

from ai_lab.client import PROTOCOLS, Provider, catalogue, packaged_providers_file, user_providers_file
from ai_lab.wire import WIRES

__all__ = ['CATALOGUE_SCHEMA_VERSION', 'document']

#: The version of the document :func:`document` returns; a consumer refuses one it does not know.
CATALOGUE_SCHEMA_VERSION = 1


def _row(row: Provider) -> dict:
    keys = {}
    if row.keyed:
        keys = {'default': key_username()} | {task: key_username(task) for task in sorted(API_KEY_TASKS)}
    return {
        'wire': row.wire,
        'protocol': WIRES[row.wire].protocol,
        'url': row.url,
        'host': row.host,
        'paths': {p: row.path(p) for p in sorted(row.protocols)},
        'endpoints': {p: row.endpoint(p) for p in sorted(row.protocols)},
        'auth': row.auth,
        'key_name': row.key_name,
        'key_usernames': keys,
        'vision': row.vision,
        'vendor': row.vendor,
        'model_wires': dict(row.model_wires),
        'models': dict(row.models),
    }


def document(*, now: datetime | None = None) -> dict:
    """The resolved catalogue with its inputs, their hashes, the ai-lab version and the generation time."""
    inputs, digest = [], hashlib.sha256()
    for path in (packaged_providers_file(), user_providers_file()):
        data = path.read_bytes() if path.is_file() else None
        digest.update(data or b'')
        inputs.append({
            'path': str(path),
            'exists': data is not None,
            'sha256': None if data is None else hashlib.sha256(data).hexdigest(),
        })  # fmt: skip
    return {
        'schema_version': CATALOGUE_SCHEMA_VERSION,
        'generated_at': (now or datetime.now(UTC)).isoformat(),
        'ai_lab_version': version('ai_lab'),  # the installed metadata; __version__.py is a build artefact
        'python': sys.executable,
        'inputs': inputs,
        'source_sha256': digest.hexdigest(),
        'protocols': sorted(PROTOCOLS),
        'providers': {name: _row(row) for name, row in sorted(catalogue().items())},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the catalogue: ``--json`` the versioned document, else one line per provider."""
    parser = argparse.ArgumentParser(prog='python -m ai_lab.providers', description=__doc__.split('\n', 1)[0])
    parser.add_argument('--json', action='store_true', help='the whole resolved catalogue as one JSON document')
    args = parser.parse_args(argv)
    doc = document()
    if args.json:
        sys.stdout.write(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
        return 0
    for name, row in doc['providers'].items():
        sys.stdout.write(f'{name:12} {row["auth"]:6} {row["wire"]:18} {" ".join(row["endpoints"].values())}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
