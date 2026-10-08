# ai-lab

Typed decisions and schema-bound responses from any model, through one interface. No runtime
dependency.

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
or a CLI binary), the environment variable holding its key, and whether it reads images. Set
`AI_LAB_PROVIDERS` to your own TOML to add or replace rows.

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

## Keys

Keys are read from the environment at call time and never stored, logged or put in a repr. On
Windows: `setx OPENAI_API_KEY "sk-..."`, then restart the shell.

## Develop

```
make verify        # ruff + unit tests (offline)
make live          # real calls: local CLIs and every provider whose key is set
```
