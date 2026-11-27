# Local llama.cpp provider (`llamacpp`)

A model served by `llama-server` on this machine: nothing leaves the box. The catalogue row
`[llamacpp]` (`src/ai_lab/providers.toml`) says where it answers (`http://127.0.0.1:8080`, OpenAI and
Anthropic Messages under `/v1`); your `serve.toml` says how this machine starts it. Claude Code reaches it
as `ccla` (claude-code-config), which runs `python -m ai_lab.serve ensure llamacpp` first.

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

## 4. Declare how to start it: `serve.toml`

In the ai-lab config root (`$AI_LAB_HOME/config`, else the platform's user config dir for `ai_lab`,
beside your own `providers.toml`). `--host`/`--port` come from the catalogue url and are refused here.

```toml
[llamacpp]
command = ["~/tools/llama.cpp/llama-server", "-m", "~/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf",
           "--alias", "qwen3.8-27b", "--jinja", "-ngl", "99", "-c", "196608", "-fa", "on",
           "-ctk", "q8_0", "-ctv", "q8_0", "--parallel", "1"]
```

- `--alias` must equal the row's `models.base` (the name Claude Code requests).
- `--jinja` is required: without the model's chat template there is no tool calling.
- `--parallel 1` gives the one slot the whole context; Claude Code's side calls queue behind it.

## 5. Start and check

```bash
python -m ai_lab.serve ensure llamacpp    # running | started; the log is serve-llamacpp.log beside serve.toml
curl -s localhost:8080/v1/messages -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}'
```

A started server is detached and keeps the model loaded until you stop it (`taskkill /IM
llama-server.exe /F`, or `pkill llama-server`).

## Another model

Add the GGUF under `~/models`, point `serve.toml` at it, and either change `models.base` in a row of your
own `providers.toml` (a user row replaces the packaged one) or add a second local row on another port.
