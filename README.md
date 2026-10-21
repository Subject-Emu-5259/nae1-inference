# nae1-inference

Inference service for **NeuralAI — Nae1**, rebuilt 2026-10-21 for the
**SFT v8 assistant release** after the original repo was deleted. The service
contract was reconstructed from the original deployment's live /health
response and `serving/tool_protocol.py` in the NeuralAI repo.

## What it serves

- Checkpoint: `sft_v8_step16000_bf16/` from
  [Subject-Emu-5259/NeuralAI-Nae1](https://huggingface.co/Subject-Emu-5259/NeuralAI-Nae1)
  (bf16 serving copy of the gated f32 release; golden-40 32.5/40 PASS,
  tool calls 36/40, guardrail PPL 59.26 on the 50-doc set)
- Engine: native PyTorch `Nae1ForCausalLM`, bf16, CPU
- Downloads the checkpoint from Hugging Face at boot into `model_cache/`

## Endpoints

| Endpoint | Auth | Notes |
| --- | --- | --- |
| `GET /health` | public | Render health check; model facts JSON |
| `POST /chat` | `Authorization: Bearer $CHAT_TOKEN` | Two shapes. API: `{messages:[{role,content}], max_tokens?, temperature?, top_p?}` → `{reply, tool_calls}`. Site protocol (public chat page): `{prompt, temperature, top_k, top_p, max_new_tokens, repetition_penalty}` → SSE-style stream of `data: {"text": <cumulative>, "done": bool}` lines ending `done: true, eos: true`. Both run the trained tool loop (max 3 rounds) |
| `POST /generate` | Bearer `$CHAT_TOKEN` | `{prompt, max_tokens?|max_new_tokens?, repetition_penalty?}` → `{text}` raw completion |
| `GET /diag` | Bearer `$DIAG_KEY` | readiness, uptime, model source |

## Tool loop

Prompts render through the checkpoint's own shipped chat template
(`tokenizer_config.json`). The generation prompt ends `NeuralAI:` with no
trailing space — the space fuses into the next ByteLevel token (4776, ' »'),
and a trailing space tokenizes off-distribution (see the model card).

Tools: `fetch` (public http(s) only, private addresses refused, result
truncated to 250 tokens), `read_file` (sandboxed to `DATA_DIR`), `search`
(no backend configured on this deployment — returns a protocol error block
the model can react to).

## Run locally

```bash
MODEL_DIR=/path/to/sft_v8_step16000 CHAT_TOKEN=dev DIAG_KEY=dev python server.py
# PORT defaults to 10000
```

## Deploy

Render web service, Docker runtime, repo branch `main`, auto-deploy on
commit. Env vars on the service: `CHAT_TOKEN`, `DIAG_KEY` (secrets, set in
the Render dashboard — never in this repo).
