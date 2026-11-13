# nae1-inference

Inference service for **NeuralAI — Nae1**, rebuilt 2026-10-21 for the
**SFT v8 assistant release** after the original repo was deleted. The
checkpoint has moved with each gated release since: v9, v10, v11, v12.2,
v13.2, and now **SFT v17 (LoRA step-9250)**, serving since 2026-11-11.
The service contract was reconstructed from the original deployment's
live /health response and `serving/tool_protocol.py` in the NeuralAI
repo.

## Live

- **Chat with NEA1:** https://neuralai-nae1.deandrewh26-d6e.workers.dev — the public chat site; this service is its inference backend
- **NeuralAI — the company:** https://neuralai.wasmer.app
- **The model:** [GitHub](https://github.com/Subject-Emu-5259/NeuralAI) · [Hugging Face](https://huggingface.co/Subject-Emu-5259/NeuralAI-Nae1)

## What it serves

- Checkpoint: `sft_v17_lora_step9250_bf16/` from
  [Subject-Emu-5259/NeuralAI-Nae1](https://huggingface.co/Subject-Emu-5259/NeuralAI-Nae1)
  (bf16 serving copy of the gated release; golden-40 35.5/40 PASS,
  tool calls 39/40, grounded answers 32/40, guardrail PPL 50.49 on
  the 50-doc set)
- Engine: native PyTorch `Nae1ForCausalLM`, bf16, CPU
- Downloads the checkpoint from Hugging Face at boot into `model_cache/`

## Serving

Primary serving is **Fabric**: this same `server.py` runs on the Mac
node and the VM node behind the Fabric switchboard
(`nea1-fabric`), which routes chat and jobs to the nodes. **Render**
(`nae1-inference`, Docker, repo branch `main`, auto-deploy on commit)
is the fallback and serves the same checkpoint. **Modal is disabled**
(billing) — `modal_app.py` stays in the repo, but the Modal service
is not running and the workspace is not pointed at it.

## Endpoints

| Endpoint | Auth | Notes |
| --- | --- | --- |
| `GET /health` | public | Health check; model facts JSON (`status`, `model`, `step`, `heldout_ppl`) |
| `GET /v1/models` | public | OpenAI-shape model list (`nea1-v13.2` — the public model id) |
| `POST /chat` | `Authorization: Bearer $CHAT_TOKEN` | `{messages:[{role,content}], max_tokens?, temperature?, top_p?, namespace?}` → SSE stream of `data: {"text": <cumulative>, "done": bool}` lines ending `done: true, eos: true`. The stream opens with a `{"node": <name>}` frame and the done frame carries the serving `model` label. Runs the trained tool loop (max 3 rounds) |
| `POST /v1/chat/completions` | Bearer `$CHAT_TOKEN` | OpenAI-compatible completions (streaming + non-streaming) |
| `POST /generate` | Bearer `$CHAT_TOKEN` | `{prompt, max_tokens?\|max_new_tokens?, repetition_penalty?}` → `{text}` raw completion |
| `POST /abort` | Bearer `$CHAT_TOKEN` | Best-effort abort of an in-flight generation |
| `GET /diag` | Bearer `$DIAG_KEY` | readiness, uptime, model source |

## Tool loop

Prompts render through the checkpoint's own shipped chat template
(`tokenizer_config.json`). The generation prompt ends `NeuralAI:` with no
trailing space — the space fuses into the next ByteLevel token (4776, ' »'),
and a trailing space tokenizes off-distribution (see the model card).

System messages never render as model speech: the server extracts them
and prepends them as a bare preamble before the conversation (the
checkpoint template has no system role — every non-user message would
otherwise render as `NeuralAI:`).

Tools: `fetch` (public http(s) only, private addresses refused, result
truncated to 250 tokens) and `calc` (deterministic arithmetic, answered
without generation). `search` and `read_file` were removed from the
callable set on 2026-11-12 — the model can no longer call them.

## Deterministic layers

Two question classes never reach generation — they are answered from
code, so they are always right:

- **§8 calc + date.** Direct arithmetic (`8473 * 96`, `8473 times 96`)
  and direct date questions (`what is today's date?`, `what is tomorrow?`)
  are answered deterministically (America/Chicago). Fail-open: anything
  the detectors don't recognize falls through to the model.
- **Grounding guard.** Fetch answers are checked against the fetched
  page text before they are returned: a garbled or empty page gets an
  honest "I couldn't read that page", a claim the page doesn't support
  is corrected from the actual page text, and a claimed page result
  with no fetch behind it is refused.

## Memory

A `namespace` field on `/chat` (`user-<id>`) enables per-user memory:
the node recalls the user's stored facts before answering and remembers
explicit facts (name, preferences, projects) after the turn. Only the
VM node reaches the memory server today — other legs fail open with no
recall and no error. Secrets (passwords, keys, card numbers) are never
stored.

## Run locally

```bash
MODEL_DIR=/path/to/sft_v17_lora_step9250_bf16 CHAT_TOKEN=<redacted> python server.py
# PORT defaults to 10000
```

## Deploy

Render web service, Docker runtime, repo branch `main`, auto-deploy on
commit. Env vars on the service: `CHAT_TOKEN`, `DIAG_KEY` (secrets, set in
the Render dashboard — never in this repo). The Fabric nodes run the
same file: the VM node via `ensure_node.sh`, the Mac node from
`~/nea1-fabric-node/server.py`.
