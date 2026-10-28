---
name: every-review-calls-only-anthropic-by-default
description: User ruling 2026-10-28 -- sharp review (interactive) and forge PR review (headless) default to anthropic via the shipped DEFAULT CONFIG (method_data/default-config.json), not method data; the roster rotation stays
created: 2026-10-28
---

# Every review calls only anthropic by default -- a default config, not method data

User ruling 2026-10-28: every lab-commons-family review -- sharp review (interactive) AND the forge PR
review (headless) -- calls ONLY the anthropic API by default. Corrected the same day: "should not be
hard-coded, but a default configuration" -- the first cut rewrote `method.json` and deleted the
roster rotation, which broke `keep-valuable-options`; it was reverted.

- The default lives in ONE place: `src/ai_lab/review/method_data/default-config.json`
  (`providers = {interactive: [anthropic], headless: [anthropic]}`), read as
  `ai_lab.review.method.DEFAULT_CONFIG`; `configured_providers` resolves project config, then it.
- Order (`plan._providers`): request > project `.claude/sharp-review.json` `providers.<mode>` >
  default config > the method's run-mode list / roster rotation. Headless `run` uses the same resolver.
- The method keeps its capability: interactive's roster rotation (`count: 2`) still runs when the default
  is emptied -- pinned by the planted control in `tests/unit/test_review_plan.py`.
- Reviewer count is `pick.count`, never the provider list. Interactive `models` gained `anthropic: opus`.
