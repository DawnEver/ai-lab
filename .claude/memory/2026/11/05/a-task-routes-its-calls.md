---
name: a-task-routes-its-calls
description: routes.toml names each task's provider chain and fall_over_on classes; the task set is derived from it; login --task T --bundle - replaces a task's slots from one TOML bundle; needs lab-commons pr/key-tasks-are-not-a-constant in the lock
created: 2026-11-05
---

# A task routes its calls

User-approved design 2026-11-05 ("key routing design"). Supersedes the rejected `fallback = ["deepseek"]`
row field of `pr/deepseek-follows-anthropic` (18c0151): providers.toml stays provider facts.

- `src/ai_lab/routes.toml` + the user's beside providers.toml (a task there replaces its namesake).
  Tasks: pr-review, optimization, session (anthropic -> deepseek), live (anthropic, openai; `fall_over_on = []`).
- Error classes, closed, in `ai_lab.client`: `ERROR_CLASSES` = quota auth rate_limit server; `error_class()`
  derives one from `ProviderError.kind` + its new `status` (5xx -> server). `ProviderError.kind` spellings are
  unchanged on purpose: dev-lab's forgereview reads `exhausted`.
- `connect(task=t)` / `connect(spec, task=t)`; a stand-in's ledger row names `fallback_from`; 529 -> rate_limit.
- Review default provider = head of the pr-review route (`review.method.default_config()`); default-config.json gone.
- Bundle: `python -m ai_lab.auth login --task <t> --bundle -`, TOML `[keys] provider = ["k1", "k2"]`.
- Hosted `live`: secret `AI_LAB_KEYS_LIVE`, setup `python -m ai_lab.auth login --task live --bundle -`.
  The rendered ci.yml still pipes `printf NAME=value` until dev-lab renders a bundle secret raw.
- DEPENDENCY ORDER: needs lab-commons `TASK_NAME` (no `API_KEY_TASKS`) and `erase_credential`; measured
  green with PYTHONPATH on that lane until uv.lock is bumped.
