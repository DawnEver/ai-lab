---
name: a-task-falls-over-from-anthropic-to-deepseek
description: providers.toml `fallback = ["deepseek"]` on the anthropic row; a TASK's call moves to deepseek (same task's slots, same role's model, anthropic wire) once every anthropic slot fails at the account level; no-task calls (live) never move; ledger rows carry fallback_from
created: 2026-11-05
---

# A task falls over from Anthropic to DeepSeek

User ruling 2026-11-05: DeepSeek is cheaper and plentiful; everywhere a task uses an Anthropic key it is the
fallback once Anthropic is used up. Also: the live part takes a second Anthropic key (slot 2).

- Chain declared once: `fallback = [...]` on a providers.toml row; `catalogue()` refuses an unknown, self or
  keyless entry. `Client.decide`/`respond` walk it (`_along_chain`); `decide_here`/`respond_here` are one provider.
- Moves only on `ACCOUNT_KINDS` (exhausted, auth, rate_limit = 429 or 529 past retries) or a daily token cap, after every
  slot failed; a per-call error (kind '') or a timeout never moves. A call with no task never moves -- the
  `live` tests connect without one, pinned by `test_the_live_tests_never_name_a_task`.
- Model: the role the anthropic model plays in its row's `models` (base when none), mapped through deepseek's
  `models` with the launcher's `[1m]` context marker stripped: opus/sonnet/haiku/base -> deepseek-v4-flash,
  fable -> deepseek-v4-pro. Protocol kept (anthropic wire) since the deepseek row serves it.
- deepseek `anthropic_path` corrected `/anthropic` -> `/anthropic/v1`: a path is where the wire's verbs hang
  (anthropic's own is `/v1`), and DeepSeek serves `/anthropic/v1/messages`. The two pins on the old URL were
  factually wrong goldens (no call had ever used it).
- Visible: a WARNING on logger `ai_lab.client` (task, from, to, kind -- never a key) and `fallback_from` on the
  ledger row (`budget.charge`), so spend per provider stays measurable. Batches never fall over.
- A missing deepseek task key re-raises the anthropic failure (its kind) naming `login deepseek --task <task>`.
