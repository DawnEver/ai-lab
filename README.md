# ai-lab

Typed decisions and schema-bound responses from any model, through one interface. Part of the
lab-commons family: its config search path and process-tree reaping are lab-commons', and so are the
lint set, the rendered project files and the verify entry point.

## Two verbs

| verb | asks | returns |
|---|---|---|
| `decide(DecisionRequest)` | closed questions: `Predicate`, `Choice`, `Score` | `Decision`: one typed answer per question, with a `Basis` |
| `respond(ResponseRequest)` | instructions + context + a JSON schema | `Response`: an object of that schema |

`Basis` says where a probability came from: `native` (the vendor returned a distribution),
`stated` (a model named an answer; probabilities are `None`, never a faked 1.0), `rule` (code).

A context is a tuple of parts: `Text`, `Fields` (named values; a rule reads them directly, a model
reads them as JSON) and `Image`.

```python
from ai_lab import Choice, DecisionRequest, Fields, Option, Predicate, Text, connect

request = DecisionRequest(
    (Text(open('knowledge/ipm.md').read()), Fields({'torque_nm': 327.0, 'magnet_temp_c': 142.0})),
    (
        Predicate('worth_fea', 'Is this design worth an expensive FEA?'),
        Choice('fidelity', 'Which fidelity next?', (Option('drop'), Option('fea2d'), Option('fea3d'))),
    ),
)
decision = connect('openai_decisions:gpt-6-luna').decide(request)
decision['fidelity'].choice, decision['fidelity'].probabilities
```

## Providers are data, wires are protocols

`connect('<provider>:<model>')` reads `providers.toml`: each row names a wire, an endpoint (a URL
or a CLI binary), the environment variable holding its key, and whether it reads images. Put a
`providers.toml` in the family config root (`$AI_LAB_HOME/config`, else the platform's user config
directory for `ai_lab`) to add rows or replace one of the same name.

| wire | decides | responds | packaged providers |
|---|---|---|---|
| `openai_decisions` | native | -- | `openai_decisions` |
| `typesafe` | native | -- | `typesafe` (Jev) |
| `openai_responses` | emulated | yes | `openai` |
| `openai_chat` | emulated | yes | `openrouter`, `deepseek`, `qwen`, `ollama` |
| `anthropic_messages` | emulated | yes | `anthropic` |
| `gemini` | emulated | yes | `gemini` |
| `claude_code` | emulated | yes | `claude_code` (local `claude`, your login) |
| `codex` | emulated | yes | `codex` (local `codex`, your login) |

A wire's verbs are read off its code. "Emulated" means the questions become one derived strict
JSON schema; the answers are then `stated`. A capability the provider lacks (a verb, images, a key)
raises `Unsupported` naming the remedy.

Local CLIs run in an empty temporary directory with every tool, MCP server and settings source
off (`claude`) or a read-only ephemeral sandbox (`codex`): they answer, they cannot act.

## Replay

```python
from ai_lab import Ledger

client = Ledger('output/ai/ledger.jsonl', mode='record').wrap(connect('anthropic:claude-sonnet-5-5'))
```

Every call is keyed by the SHA-256 of verb, client and canonical request (images by digest). `record`
serves hits and records misses; `replay` serves hits and raises `ReplayMiss` on a miss, so a
replayed run never reaches a model.

`Rules({...})` is a decider made of code over `Fields`: the offline baseline any model decider has
to beat.

## Effort, usage and cost

`connect('openai:gpt-x@high')` sets a reasoning effort; it is part of the client's name, so the
ledger keeps two efforts apart. A wire without an effort hook refuses one (`openai_decisions` takes
none -- the endpoint rejects the parameter).

Every `Decision` and `Response` carries the vendor's `usage` (input, cached, output, reasoning
tokens) where the wire reads it, and the ledger records it with the call's time.

Cost is derived, never stored: tokens times the rate in force at the call. Rates come from public
catalogs (models.dev, LiteLLM, Portkey; OpenRouter only for the `openrouter` vendor -- a reseller's
route price is not the vendor's), cross-checked to their median and flagged DISPUTED beyond 5%,
kept as dated snapshots in the user cache and refreshed at most daily. A provider row's `vendor`
says who bills it (`""` = unmetered, e.g. a CLI on a subscription). A row in your own `prices.toml`
beside `providers.toml` (US dollars per million tokens, keyed `provider:model`) is a contract rate
and wins:

```toml
["openai_decisions:gpt-6-luna"]
input = 0.10
cached_input = 0.01
output = 0.50
```

- `python -m ai_lab.prices CLIENT [...]` -- each catalog's quote, the consensus, agreement.
- `python -m ai_lab.report LEDGER [...]` -- calls, latency, tokens and cost per `provider:model[@effort]`.
- `python -m ai_lab.evaluate DATASET --decider SPEC [...]` -- every decider on the same labelled
  requests (`{"request": ..., "labels": {...}}` per line): AUC, precision, recall, latency, tokens,
  cost. With `--ledger`, known calls are served from it -- a recorded run is a free benchmark.

## Keys

Keys are read from the environment at call time and never stored, logged or put in a repr. On
Windows: `setx OPENAI_API_KEY "sk-..."`, then restart the shell.

## Develop

```
uv pip install -e ".[dev]"     # the family dev kit (lab-commons[dev]), pytest, ruff
make verify                    # python -m lab_commons.dev.verify: ruff, the suite, hook wiring
python -m pytest -m live       # real calls: local CLIs and every provider whose key is set
```

`.gitignore`, `.gitattributes`, `.rgignore`, `.pre-commit-config.yaml` and `Makefile` are RENDERED from
the family bases (`tests/architecture/_famconfig.py` holds ai-lab's delta, empty today): a hand edit
reds `tests/architecture/test_the_family_config_is_rendered.py`.
