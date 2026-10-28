---
name: every-review-calls-only-anthropic-by-default
description: User ruling 2026-10-28 -- sharp review (interactive) and forge PR review (headless) both default to providers ["anthropic"]; the available-roster rotation is deleted
created: 2026-10-28
---

# Every review calls only anthropic by default

User ruling 2026-10-28: every lab-commons-family review -- sharp review (interactive) AND the forge PR
review (headless) -- calls ONLY the anthropic API by default.

- `method.json` `run_modes.interactive.providers` was a roster rotation over the available providers
  (`count: 2`); it is now `["anthropic"]`, like headless. Interactive `models` keys `anthropic: opus`.
- Reviewer count is `pick.count` (profiles), never the provider list: two reviewers, both anthropic.
- The roster branch of `plan._providers` was dead once no mode used a dict spec and is deleted.
- Kept: `requested` and config `providers.<run mode>` still override -- other vendors are opt-in, explicitly.
- Pinned by `tests/unit/test_review_plan.py::test_every_run_mode_calls_only_anthropic_by_default`.
