---
name: local-llama-provider
description: llamacpp catalogue row and ai_lab.serve start a local llama-server on demand; measured context sizing and the Qwen template patch
title: llamacpp local provider and ai_lab.serve
created: 2026-11-27
tags: [providers, llama.cpp, local, serve]
---

# llamacpp local provider (#67, merged 2026-11-27)

- `[llamacpp]` row: `http://127.0.0.1:8080`, openai and anthropic paths `/v1`, `auth = "local"`,
  `models.base = "qwen3.8-27b"`. `python -m ai_lab.serve ensure <provider>` starts the server from the
  user's `serve.toml` (argv; `--host/--port` derived from the row url and refused in the file), waits for
  `/health`, log `serve-<provider>.log` beside it. Setup doc: `docs-src/local-llama.md` (docs go in
  `docs-src/`; `docs/` is not in layout.toml).
- Install the CUDA release zip (cuda-13.x for Blackwell), not winget (Vulkan, far slower prefill).
- Qwen3.8-27B is hybrid attention (16 of 64 layers keep KV, 4 heads x 256), so long context fits 24 GB.
  Measured RTX PRO 4000 Blackwell, Q4_K_M, ~27k prompt: 196608 q8/q8 22.7 GB 1266 pp / 29.5 tg tok/s
  (default); 262144 q8/q4 23.2 GB 1214 / 26.0; 262144 q8/q8 spills to shared memory, ~45 tok/s prefill.
  A full card + tens-of-tok/s prefill = Windows silent VRAM paging.
- Claude Code sends system text after the first message; Qwen's template raises (HTTP 500). Patched
  template (late system -> system turn) via `--chat-template-file`; recipe in the doc.
- Consumers: claude-code-config `ccla`/`codla`; local-only sessions are filtered there.
