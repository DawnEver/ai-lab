---
name: a-task-owns-its-key
description: login <provider> --task <task> stores api-key-<task> under the provider host; connect(spec, task=) sends only it, no slots, no default fallback; needs lab-commons af6bb89 (lane/task-api-keys) in the lock
created: 2026-10-19
---

# A task owns its key

User ruling 2026-10-19: each task gets its own AI API key, in the one existing scheme.

- The username and the declared tasks live in `lab_commons.credentials` (`API_KEY_USERNAME`,
  `API_KEY_TASKS` = pr-review, optimization, `task_username`, `api_key`); ai-lab derives from them.
  `slot_username` now builds on `API_KEY_USERNAME`, and `_slot_key` uses `credential_for(username=)`,
  retiring the private `_credential`/`_why_no_credential` import.
- `auth login <provider> --task T` (no `--slot` with it), `status` gains a tasks column.
- `connect(spec, task=T)`: one key, reported as slot 1; missing -> `Unsupported` naming
  `login <provider> --task T`, even with the default stored.
- DEPENDENCY ORDER: this commit imports names only lab-commons `af6bb89` has. Until that lands and
  `uv.lock` is bumped to it, `tests/architecture/test_ai_lab_adopts_the_shared_registry.py` (a subprocess
  on the installed lab-commons) reds on the import. Measured green with PYTHONPATH on the lab-commons lane.
- Follow-up in motronics: the `turing` decider passes `task='optimization'` to `connect`.
