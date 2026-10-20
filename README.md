# ai-lab

Typed decisions and schema-bound responses from any model, through one interface. Part of the
lab-commons family: its config search path and its key store are lab-commons', and so are the lint set, the rendered project files and the verify entry point.

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
decision = connect('openai:gpt-6-luna').decide(request)
decision['fidelity'].choice, decision['fidelity'].probabilities
```

## Providers are data, wires are protocols

`providers.toml` is the ONE provider catalogue. `connect('<provider>:<model>[@effort]')` reads it; a
host launcher reads the same table through `python -m ai_lab.providers --json`. A row declares:

| field | means |
|---|---|
| `wire` | the default wire (a module in `ai_lab/wire`); each wire speaks one protocol |
| `url` | the origin (scheme and host, no path); the key is stored under its host |
| `openai_path`, `anthropic_path`, `gemini_path`, `typesafe_path` | where each protocol the account serves lives; `""` is not served |
| `model_wires` | `{model = wire}`: a model of the same account served by another wire |
| `models` | the Claude-role models `base` (required, the default of the rest), `fable`, `opus`, `sonnet`, `haiku`, `subagent`, `codex` |
| `auth` | `api` (a keyed account billed per token) or `local` (a keyless server you run); a missing or unknown one is refused |
| `key_name` | the conventional name of its key (`DEEPSEEK_API_KEY`), `""` for none; only `auth login --from-file` reads it |
| `vision`, `vendor` | whether it reads images; who bills it, as the price catalogs name it (`""` = unmetered) |

Put a `providers.toml` in the family config root (`$AI_LAB_HOME/config`, else the platform's user
config directory for `ai_lab`) to add rows or replace one of the same name.

One provider is ONE ACCOUNT OF ONE VENDOR. A second door into an account is never a second row: a
door only some models use is a `model_wires` entry (`openai:gpt-6-luna` goes to OpenAI's
`/decisions`, any other `openai:<model>` to `/responses`), and a second protocol is a path of the same
row -- `deepseek` serves `openai_path = "/v1"` and `anthropic_path = "/anthropic"`, and
`connect('deepseek:<model>', protocol='anthropic')` picks the wire of that protocol.

| wire | protocol | decides | responds (schema) | packaged providers |
|---|---|---|---|---|
| `openai_decisions` | openai | native | -- | `openai` (its `model_wires`: `gpt-6-luna`) |
| `typesafe` | typesafe | native | -- | `typesafe` (Jev) |
| `openai_responses` | openai | emulated | yes, enforced | `openai` |
| `openai_chat` | openai | emulated | yes, enforced (`json_schema`) | `openrouter`, `qwen`, `ollama` |
| `openai_chat_json` | openai | emulated | yes, NOT enforced (`json_object`; the schema rides in the prompt) | `deepseek` |
| `anthropic_messages` | anthropic | emulated | yes, enforced (structured output) | `anthropic` (and `deepseek` with `protocol='anthropic'`) |
| `gemini` | gemini | emulated | yes, enforced | `gemini` |

A wire's verbs are read off its code. "Emulated" means the questions become one derived JSON schema
and the answers are `stated`; on a wire that does not enforce the schema each answer is still checked
against its declared options. A capability the provider lacks (a verb, images, a key, a protocol)
raises `Unsupported` naming the remedy.

ai-lab is the pay-per-token API layer ONLY. The logged-in vendor CLIs were retired as providers on
2026-10-20: a subscription is used inside a host session (Claude Code, Codex, or fabric's agent mode),
never through ai-lab. Every retired spelling -- providers, wires, row fields, auth kinds -- is listed
with its replacement in `ai_lab.retired` and refused by name, never aliased.

## The catalogue as a document: `python -m ai_lab.providers --json`

One JSON document, `ai_lab.providers.CATALOGUE_SCHEMA_VERSION` (1): `schema_version`, `generated_at`,
`ai_lab_version`, `python` (the interpreter that produced it), `inputs` (`[{path, exists, sha256}]`:
the packaged file, then the user file), `source_sha256` (the sha256 of the packaged bytes followed by
the user bytes, none when absent -- recomputable from the two paths without Python), `protocols`, and
`providers`: per row `wire`, `protocol`, `url`, `host`, `paths`, `endpoints` (url + path per
protocol), `auth`, `key_name`, `key_usernames` (`{"default": "api-key", "<task>": "api-key-<task>"}`,
slot 1; never a key), `vision`, `vendor`, `model_wires`, `models`. A host launcher's setup cache is
built from it; without arguments the command prints one line per provider.

## One call from any language: `python -m ai_lab.call` (protocol version 1)

A process in another language (a Node host harness) makes one call by writing ONE JSON request to
stdin and reading ONE JSON reply from stdout. Version 1 of the contract, `ai_lab.call.CALL_SCHEMA_VERSION`:

```jsonc
// request; any field not listed here is refused
{"verb": "respond" | "decide", "provider": "anthropic", "model": "claude-sonnet-5-5",
 "task": "pr-review",            // optional: walk that task's key slots only
 "effort": "high",               // optional: refused by a wire with no effort hook
 "instructions": "...",          // decide: becomes the first text of the context
 "context": [{"kind": "text", "text": "..."}, {"kind": "fields", "values": {"n": 1}}],
 "schema": {"type": "object"},   // respond only, required
 "questions": [{"kind": "predicate", "name": "p", "instructions": "..."}],  // decide only, required
 "images": [{"mime": "image/png", "data": "<base64>"}]}                      // optional

// reply
{"schema_version": 1, "ok": true,
 "kind": "" | "exhausted" | "auth" | "rate_limit" | "unsupported" | "timeout",
 "result": {...} | null,         // respond: the schema object; decide: {"basis", "answers": {name: answer}}
 "usage": {"input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "cost_usd"} | null,
 "key_slot": 1, "error": ""}
```

A provider failure is a REPLY, not a crash: `ok` is false, `kind` classifies it (`""` when the failure
is not one of the named kinds; `unsupported` for a missing key, provider or capability), `error`
names it without a key, and the exit status is 0. Only a malformed request (not JSON, an unknown or
missing field, a wrong type, an undeclared task) exits 2, with the reason on stderr. `--ledger PATH`
records every call with its usage and key slot through the ledger (record mode: a request already in
the file is served from it).

## A headless code review: `python -m ai_lab.review`

```
python -m ai_lab.review --method <sharp-review method dir> --range main...HEAD --model <model id>
                        [--providers anthropic ...] [--repo <path>]
```

The review METHOD is data owned by sharp-review (cc-market `sharp-review/method/`): profiles, prompt
templates, the finding schema and the merge rules under one `schema_version`. `--method` names that
directory and it is read in place; ai-lab keeps no copy. The run follows `run_modes.headless`: it picks
`pick.count` diff-context profiles by weight, renders each prompt over `git diff <range>`, assigns
reviewer i the provider `providers[i mod n]` (default `["anthropic"]`), and makes each call as a
`respond` under task `pr-review` whose schema is `{"findings": [<finding schema>]}`. `--model` serves
every provider the run mode's `models` does not name (the method names none headless).

stdout is ONE findings JSON document, the method's `findings-output.schema.json` (version 1), exit 0
whatever the reviewers did. Each reviewer's `status`: `ok` (a findings array; `[]` is a clean pass);
the failed call's kind (`exhausted`, `auth`, `rate_limit`, `timeout`, `unsupported`); `unavailable` (the
provider is not in the catalogue -- never substituted -- or the failure was unclassified); `skipped` (an
empty diff, no call made). A refused invocation -- no `method.json`, an unknown `schema_version`, a range
git cannot diff -- exits 2 naming it on stderr with nothing on stdout. The rules the data cannot state
(pick, templating, status precedence, merge) are pinned by the method's `conformance/` fixtures, which
this test suite reads in place (`SHARP_REVIEW_METHOD`, default the cc-market checkout) and the Node
executor runs too; the two share no code.

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
ledger keeps two efforts apart. A wire without an effort hook refuses one (`openai:gpt-6-luna` takes
none -- the `/decisions` endpoint rejects the parameter).

Every `Decision` and `Response` carries the vendor's `usage` (input, cached, output, reasoning
tokens) where the wire reads it, and the ledger records it with the call's time.

Cost is derived, never stored: tokens times the rate in force at the call. Rates come from public
catalogs (models.dev, LiteLLM, Portkey; OpenRouter only for the `openrouter` vendor -- a reseller's
route price is not the vendor's), cross-checked to their median and flagged DISPUTED beyond 5%,
kept as dated snapshots in the user cache and refreshed at most daily. A provider row's `vendor`
says who bills it (`""` = unmetered, e.g. a keyless local server). A row in your own `prices.toml`
beside `providers.toml` (US dollars per million tokens, keyed `provider:model`) is a contract rate
and wins:

```toml
["openai:gpt-6-luna"]
input = 0.10
cached_input = 0.01
output = 0.50
```

- `python -m ai_lab.providers [--json]` -- the resolved catalogue (see above).
- `python -m ai_lab.call [--ledger PATH]` -- one call as JSON on stdin/stdout (see above).
- `python -m ai_lab.prices CLIENT [...]` -- each catalog's quote, the consensus, agreement.
- `python -m ai_lab.report LEDGER [...]` -- calls, latency, tokens and cost per `provider:model[@effort]`.
- `python -m ai_lab.evaluate DATASET --decider SPEC [...]` -- every decider on the same labelled
  requests (`{"request": ..., "labels": {...}}` per line): AUC, precision, recall, latency, tokens,
  cost. With `--ledger`, known calls are served from it -- a recorded run is a free benchmark.

## Keys

Keys live in the git credential store and nowhere else, read and written through
`lab_commons.credentials` -- the same store that holds your forge token. A key is stored under the
`https` host of its provider's `url`; no environment variable is read, on any platform. Calls read the
key when they need it and never log it or put it in a repr. When a key is missing, the call is refused
and the error names the command that stores it.

An API provider may hold several keys as ordered FALLBACK slots: usernames `api-key`, `api-key-2`,
`api-key-3`, ... under the same host, read up to the first missing slot. A call uses slot 1 and moves
to the next slot only on an ACCOUNT failure (`ai_lab.errors.ACCOUNT_KINDS`: exhausted balance or
quota, a refused key, a rate limit past its retries); a timeout or any other failure would repeat on
another key and is raised as it is. When every slot fails the error lists each slot's failure kind,
never a key. The slot that served a call is the `key_slot` of its `Decision`/`Response` and of its
ledger entry. Failures are classified into `ProviderError.kind` (`ai_lab.errors.ERROR_KINDS`:
`exhausted`, `auth`, `rate_limit`, `timeout`; `""` otherwise) for every wire.

A TASK owns its own keys: usernames `api-key-<task>`, `api-key-<task>-2`, ... under the same host, for a
task declared in `lab_commons.credentials.API_KEY_TASKS`. `connect(spec, task=...)` walks that task's
slots in the same fall-over order, and only those -- never another task's, never the default's; a task
with no slot 1 is refused by name. `-s`, `--slot` and `--seat` are one argument.

EVERY TASK AND SLOT GETS ITS OWN KEY. `login` refuses a key another credential of the same host already
holds -- another task, another slot, or the default -- naming that credential and the key's sha256
fingerprint (never the key); re-storing the identical key into the same credential changes nothing.
`status` lists, after the per-provider lines, every group of credentials on one host that share a key.

```
python -m ai_lab.auth login openai                 # prompts without echo (or reads one line from stdin)
python -m ai_lab.auth login openai --slot 2        # a fallback key, stored as api-key-2
python -m ai_lab.auth login anthropic --task pr-review        # a task's own key, stored as api-key-pr-review
python -m ai_lab.auth login anthropic --task pr-review -s 2   # its backup, stored as api-key-pr-review-2
python -m ai_lab.auth login --from-file keys.env   # NAME=value, export NAME=value, or NAME value lines
ssh box python -m ai_lab.auth login --from-file - < keys.env   # the same file over stdin
python -m ai_lab.auth status                       # auth kind, key slots, task slots, shared keys; never a key
```

A key file maps names to providers through each row's `key_name`. Unknown names are reported by name
and skipped. Where the key is persisted depends on the configured `credential.helper`: Git Credential
Manager on Windows, and on Linux a keychain or `git config --global credential.helper store` with
`~/.git-credentials` set to mode 600.

A user `providers.toml` that still uses a retired field is refused naming its replacement
(`ai_lab.retired`); the key itself goes into the store with `auth login`.

## Develop

```
uv pip install -e ".[dev]"     # the family dev kit (lab-commons[dev]), pytest, ruff
make verify                    # python -m lab_commons.dev.verify: ruff, the suite, hook wiring
python -m pytest -m live       # real calls: every provider whose key is stored
```

`.gitignore`, `.gitattributes`, `.rgignore` and `Makefile` are RENDERED from
the family bases (`tests/architecture/_famconfig.py` holds ai-lab's delta, empty today): a hand edit
reds `tests/architecture/test_the_family_config_is_rendered.py`.
