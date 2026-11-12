---
name: gemini-decisions-wire-and-the-decision-work-map
description: 2026-11-12 -- gemini-decision-1 (Google Decisions API) is a gemini_decisions wire of the gemini row, with daily_calls = 10000 enforced (CallCapReached); the live eval on the T3 c8eb389bf6 gate set is BLOCKED by HTTP 402 (prepaid credits depleted) after 1 unbilled call; plus a map of the decision work across ai-lab and motronics
created: 2026-11-12
---

# Gemini Decisions wire, and where the decision work stands

User request 2026-11-12: add Gemini's new decision API, test how good it is (10,000 calls/day), inventory the
existing decision work. Branch `pr/gemini-provider` (on `pr/one-price-source`, the tip of the local stack
one-spend-cap -> task-budgets -> task-budget-approval -> one-price-source, which already contains trunk).

## What Google ships (aistudio.google.com/docs/decisions-api, /rate-limits, /pricing, read 2026-11-12)

- `POST https://generativelanguage.googleapis.com/v1beta/decisions`, header `x-goog-api-key`, model
  `gemini-decision-1`. Body `{model, input, questions: {name: {type, instructions, criteria}}}`. Types are `noul`,
  `choice` and `score`, the same shape as Jev's (typesafe). `input` is text or content blocks
  (`{type:image, data, mime_type}`).
- Reply `{answers: {name: ...}, usage: {total_input_tokens, total_output_tokens: 0}}`. A safety block is an
  HTTP 400, not an answer.
- Free tier: 200 RPM, 50k TPM, 10,000 RPD, $0. Tier 1: 50,000 RPD. Paid: $0.10 per 1M input tokens, and there
  is no output. No Batch API.

## What landed (pr/gemini-provider)

- `wire/noul.py` is the ONE translator of the noul family. `typesafe` and the new `gemini_decisions` both
  use it.
- The gemini row gets `model_wires = {"gemini-decision-1" = "gemini_decisions"}` (one account, one row).
  `usage.decisions_usage` reads the input-only usage.
- A provider row may declare `daily_calls` (requests per UTC day; 0 means none). The gemini row declares
  10000. `budget.require_call` counts today's ledger rows and raises `CallCapReached` (a `SpendCapReached`
  at provider scope, so no key slot is tried) before sending. Caveat: it counts only answered (charged)
  calls, while Google also counts errored requests and resets at Pacific midnight, not UTC.
- Tests: `tests/unit/test_gemini_decisions.py` (shapes, routing, quota refusal unsent, a negative quota is
  refused). The `test_error_kinds` coverage now includes the new wire. `tests/unit` is green apart from the
  env check `hooks-installed`.

## Live evaluation: BLOCKED (billing)

- The key is stored for task `optimization` (slot 1). The machine file has NO `[ai_lab.spend_cap] gemini`
  row, so the eval ran with its own machine file (`gemini = inf`, under `output/logs/26/11/12/gemini-eval/`).
- The first call returned **HTTP 402 "Your prepayment credits are depleted"**. The key's Google project is on
  prepaid billing with no credit, not the free tier. 1 call, 0 tokens, $0. Not retried.
- The `optimization` route tried to fall over to anthropic and deepseek, and both were refused before
  sending (no ledger row). A gemini eval must not run under a task whose chain silently swaps the vendor
  being measured.
- To unblock: top up the project's prepayment credits, or use a key from a free-tier project. Declare
  `[ai_lab.spend_cap] gemini` in the machine file. Then rerun
  `output/logs/26/11/12/gemini-eval/eval_gate.py` (cached and capped by argv). Each answer is cached under
  `run/gen`.

## The benchmark it will be scored on

- R1's method (motronics `turing/campaign/replay.py`). Labels are Deb improvement pooled over every run of
  the case. Only COMPLETE batches count (unbiased recall). Metrics are recall / precision / BA, plus AUC for
  probabilities. The rule is the seed-held-out 3-feature logistic.
- Set: the local T3 at c8eb389bf6 (motronics `output/logs/26/10/24/campaign/plan-c8eb389bf6/runs`), with
  105 complete gate batches and 2468 candidates (gate-deepseek and gate-haiku on synrm/pma, gate-haiku on
  Ferrari). That is 105 Gemini calls. For R1's 3dfbac numbers (deepseek BA 0.68 vs rule 0.72; haiku 0.59
  vs 0.68) the raw rows are not on this box.

## The decision work, mapped

- ai-lab verbs:
  - `Client.decide(DecisionRequest)` answers Predicate/Choice/Score natively (openai_decisions for
    gpt-6-luna, typesafe for Jev, and now gemini_decisions) or by emulation over a schema wire.
    `Client.respond` is the steer path.
  - `ai_lab.evaluate` scores a decider on a labelled dataset.
- ai-lab money:
  - The money spend cap per provider and month, task allotments and approvals: the local stack.
  - `prices.py` is the one price source (catalog consensus plus the user's `prices.toml`).
- motronics `turing/decision.py`:
  - `DecisionRouter` (gate = one Predicate per candidate; rank = Score; fidelity = Choice) and
    `RandomRouter`.
  - `DecisionSteer` (respond, explore/keep/exploit).
  - The trace goes to `decision_dataset`. `rule.py` is the model-free rule.
- motronics findings:
  - ZDT 10/09: luna@high had the best AUC, 0.88.
  - R1 10/24: the rule matches or beats the AI gate.
  - R0 11/05-06: rank-haiku is the only consistent synrm gain, gate-haiku the only Ferrari gain;
    gate-deepseek is harmful; steer is null or harmful.
  - The terra arms never answered (the null error field, fixed by 316775b), then hit the token caps.
