# Local llama.cpp provider (`llamacpp`)

A model served by `llama-server` on this machine: nothing leaves the box. The catalogue row
`[llamacpp]` (`src/ai_lab/providers.toml`) says where it answers (`http://127.0.0.1:8080`, OpenAI and
Anthropic Messages under `/v1`); your `serve.toml` says how this machine starts it. Claude Code reaches it
as `ccla` and Codex as `codla` (claude-code-config); both run `python -m ai_lab.serve ensure llamacpp`
first. Codex speaks the Responses API, which llama-server serves under the same `/v1`.

## 1. Install llama.cpp (CUDA build)

Take the official release zip, not winget: the winget package is the Vulkan build, whose prompt
processing is several times slower on NVIDIA. Blackwell (RTX 50xx / RTX PRO) needs CUDA 12.8+, so take
the `cuda-13.x` pair.

```bash
T=b12816   # any recent tag: https://github.com/ggml-org/llama.cpp/releases
D=~/tools/llama.cpp; mkdir -p $D
for f in llama-$T-bin-win-cuda-13.5-x64.zip cudart-llama-bin-win-cuda-13.5-x64.zip; do
  curl -sSLO https://github.com/ggml-org/llama.cpp/releases/download/$T/$f && unzip -oq $f -d $D
done
$D/llama-server --list-devices    # must list CUDA0
```

Linux/macOS: the `ubuntu-cuda` / `macos-arm64` zips, or `brew install llama.cpp`.

## 2. Download a model into `~/models`

Keep weights outside any cloud-synced folder.

```bash
mkdir -p ~/models/qwen3.8-27b && cd ~/models/qwen3.8-27b
curl -sSL -C - -O https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf
```

## 3. Size the context

The KV cache, not the weights, bounds the context. Per token it is
`2 x full-attention layers x KV heads x head_dim x bytes`. Qwen3.8-27B is hybrid (Gated DeltaNet):
only 16 of 64 layers keep KV (4 heads x 256), about 64 KB/token at f16. That is what makes a long
context fit; a dense 27B (64 KV layers) needs four times as much and cannot pass ~100k on 24 GB.

Measured on an RTX PRO 4000 Blackwell (24 GB), Q4_K_M weights (16.6 GB), flash attention, one slot,
~27k-token prompt (2026-11-27, b12816):

| `-c` | `-ctk`/`-ctv` | VRAM | prefill | generate | verdict |
|---|---|---|---|---|---|
| 196608 | q8_0 / q8_0 | 22.7 GB | 1266 tok/s | 29.5 tok/s | default |
| 262144 | q8_0 / q4_0 | 23.2 GB | 1214 tok/s | 26.0 tok/s | works, ~1 GB headroom |
| 262144 | q8_0 / q8_0 | 24.0 GB | ~45 tok/s | - | spills into shared memory: unusable |

262144 is the model's native maximum. Watch for the spill: Windows does not fail an over-allocation,
it silently pages VRAM to system RAM and prefill drops ~30x. If `nvidia-smi` shows the card full and
prefill is in the tens of tok/s, lower `-c` or the V cache type. Other GPU users (a browser, a second
model) eat the same headroom.

Other models on the same card (2026-11-28, b12816, same ~27k-token prompt):

| model | layout | config | VRAM | prefill | generate |
|---|---|---|---|---|---|
| Muse Glimmer 30B, UD-Q4_K_XL 15.9 GB | dense; 13 of 52 layers full attention (2 KV heads), the rest a 2048 sliding window: ~6.5 KB/token at q8; native 131k; thinks first | 131072 | 15.9 GB | 1433 tok/s | 31.0 tok/s |
| | | 655360, YaRN x5 (`rope-scaling yarn`, `rope-scale 5`, `yarn-orig-ctx 131072`) | 20.5 GB | 1426 tok/s | 30.7 tok/s |
| Qwen3-Coder-30B-A3B, UD-Q4_K_XL 17.7 GB | MoE, 3B active; 48 full-attention layers (4 KV heads): ~48 KB/token at q8; native 262k | 65536, all on GPU | 20.4 GB | 3060 tok/s | 60.1 tok/s |
| | | 196608, `n-cpu-moe 14` | 22.7 GB | 1105 tok/s | 47.2 tok/s |
| | | 262144, `n-cpu-moe 24` | 22.9 GB | 816 tok/s | 40.4 tok/s |

Muse at 640k recalled a fact placed at the very start of a 499k-token prompt (prefill 752 tok/s, 11 min;
generation 12.2 tok/s that deep), so its YaRN context is usable, not only allocatable. 1M (YaRN x8) loads at
23.9 GB, on the spill edge. Qwen3.8-27B (~32 KB/token at q8 plus 16.6 GB of weights) and Qwen3-Coder
(~48 KB/token) cannot pass ~256k on 24 GB.

## 4. Declare the models: `serve.toml`

In the ai-lab config root (`$AI_LAB_HOME/config`, else the platform's user config dir for `ai_lab`,
beside your own `providers.toml`). It is the ONE source: `ai_lab.serve` renders llama-server's router
preset (`serve-llamacpp.ini`, INI) from it on every `ensure`; never edit the INI. One table per model holds
its llama-server options, keys as the long option without `--`; `shared` holds what all of them share.
`--host`, `--port` and `--models-preset` are set by `ai_lab.serve` and refused in `command`.

```toml
[llamacpp]
command = ["~/tools/llama.cpp/llama-server", "--models-max", "1"]
default = "qwen3.8-27b"

[llamacpp.shared]
jinja = true          # required: without the model's chat template there is no tool calling
flash-attn = "on"
n-gpu-layers = 99
parallel = 1          # one slot gets the whole context; Claude Code's side calls queue behind it

[llamacpp.models."qwen3.8-27b"]
model = "~/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
ctx-size = 196608
cache-type-k = "q8_0"
cache-type-v = "q8_0"
chat-template-file = "~/models/qwen3.8-27b/chat_template.claude-code.jinja"

[llamacpp.models."muse-glimmer-30b"]
model = "~/models/muse-glimmer-30b/Muse-Glimmer-30B-UD-Q4_K_XL.gguf"
ctx-size = 655360
cache-type-k = "q8_0"
cache-type-v = "q8_0"
rope-scaling = "yarn"
rope-scale = 5
yarn-orig-ctx = 131072

[llamacpp.models."qwen3-coder-30b-a3b"]
model = "~/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
ctx-size = 196608
cache-type-k = "q8_0"
cache-type-v = "q8_0"
n-cpu-moe = 14
```

- `--models-max 1`: one model at a time on a 24 GB card; asking for another swaps it in (~17 s).
- A model is named by its table; `ccla -l <name>` / `codla -l <name>` picks one (a prefix, a contained
  part or a near typo is enough), a bare `-l` lists them, none uses `default`.
- `chat-template-file` (Qwen3.8 only): Claude Code sends system text after the first message, and Qwen's
  template refuses that (`System message must be at the beginning`, HTTP 500). Extract the model's
  template once and render a late system message as a system turn instead:

  ```bash
  cd ~/models/qwen3.8-27b
  curl -s "localhost:8080/props?model=qwen3.8-27b" | python -c "import sys,json; t=json.load(sys.stdin)['chat_template'];   t=t.replace(\"{{- raise_exception('System message must be at the beginning.') }}\",   \"{{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\");   open('chat_template.claude-code.jinja','w',encoding='utf-8',newline='').write(t)"
  ```

  Muse Glimmer and Qwen3-Coder accept a late system message as shipped.

## 5. Start and check

```bash
python -m ai_lab.serve models llamacpp                      # the models; * marks the default
python -m ai_lab.serve ensure llamacpp --model qwen3.8-27b  # running | started, that model loaded; log: serve-llamacpp.log
curl -s localhost:8080/v1/messages -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}'
```

A launcher (`ccla`, `codla`) passes `--holder <its pid>` and runs `python -m ai_lab.serve release llamacpp
--holder <pid>` when its session ends: the server ai_lab.serve started stops once no live session holds it,
so the model's VRAM is freed with the last session. A session killed with its window never releases, so a
detached `ai_lab.serve watch` checks the holders every 5 s and stops the server once none is alive. A server you started yourself (or `ensure` without
`--holder`) is never stopped; stop it with `taskkill /IM llama-server.exe /F` or `pkill llama-server`.

## Another model

Download the GGUF under `~/models`, add a `[llamacpp.models."<name>"]` table, and size it as in section 3:
read the base model's `config.json` (full-attention layers, KV heads, head_dim, native context), then
measure VRAM and prefill at the context you want. Check that a tool call comes back as `tool_use` and that
a late system message is accepted before using it through `ccla -l <name>`.
