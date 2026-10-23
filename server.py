#!/usr/bin/env python3
"""NeuralAI — Nae1 inference service (rebuilt 2026-10-21 for the SFT v8 release).

Serves the SFT v12.2 LoRA assistant checkpoint (step-12000, adapters merged)
natively in PyTorch bf16
on CPU. Endpoints:

  GET  /health    public; deployment health + model facts (shape kept from
                  the original service)
  POST /chat      Bearer CHAT_TOKEN; {messages:[{role,content}], ...}
                  runs the trained tool loop (search/fetch/read_file)
  POST /generate  Bearer CHAT_TOKEN; {prompt, ...} raw completion
  GET  /diag      Bearer DIAG_KEY; runtime diagnostics

The chat prompt is rendered with the checkpoint's own shipped chat template
(tokenizer_config.json) — a single source of truth. Its generation prompt
ends "NeuralAI:" with NO trailing space: the space fuses with the following
token in ByteLevel BPE (id 4776 for ' »'), and a trailing space would
tokenize off-distribution. See the model card's SFT section.

Tool protocol helpers (call detection, result blocks, token-budget
truncation, transcript GC) follow serving/tool_protocol.py from the
NeuralAI repo, which survived the original service.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import re
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from nae1_model import Nae1ForCausalLM  # noqa: E402
from tokenizers import Tokenizer, decoders, pre_tokenizers  # noqa: E402

# ---------------------------------------------------------------- config
PORT = int(os.environ.get("PORT", "10000"))
CHAT_TOKEN = os.environ.get("CHAT_TOKEN", "")
DIAG_KEY = os.environ.get("DIAG_KEY", "")
MODEL_DIR = os.environ.get("MODEL_DIR", "")  # local override (tests)
HF_REPO = os.environ.get("HF_REPO", "Subject-Emu-5259/NeuralAI-Nae1")
HF_SUBDIR = os.environ.get("HF_SUBDIR", "sft_v12_2_lora_step12000_bf16")
DATA_DIR = Path(os.environ.get("DATA_DIR", str(HERE / "data")))
EOS_ID = 3
N_CTX = 2048
CTX_RESERVE = 256
MAX_ROUNDS = 3
MAX_TOOL_TOKENS = 250
TOOLS = ("search", "fetch", "read_file")
MODEL_LABEL = "nae1-sft-v12-2-lora-step12000 (bf16)"
STEP = 12000
# 50-doc guardrail set (fluency_docs_20261017), n_ctx 512 — the only PPL
# measured for this checkpoint; see the model card for the protocol note.
GUARDRAIL_PPL = 48.1
# Device/dtype are env-driven: Render keeps the defaults (cpu, bfloat16);
# the Modal deployment sets DEVICE=cuda, DTYPE=float32 (T4 has no bf16
# tensor cores; a 100M model in fp32 is still far faster than needed).
DEVICE = os.environ.get("DEVICE", "cpu")
_DTYPE_NAME = os.environ.get("DTYPE", "bfloat16")
DTYPE = {"bfloat16": torch.bfloat16, "float32": torch.float32,
         "float16": torch.float16}[_DTYPE_NAME]

STATE = {"ready": False, "error": None, "started": time.time()}
MODEL = None
TOK = None
CHAT_TEMPLATE = None
GEN_LOCK = threading.Lock()

_CALL_RE = re.compile(
    r'^\s*»\s*(?P<name>[a-z_]+)\(\s*(?P<arg>"(?:[^"\\]|\\.)*")\s*\)\s*$', re.M)
_BLOCK_RE = re.compile(r'<<tool name="(?P<name>[^"]+)" query="(?P<q>[^"]*)">>.*?<</tool>>', re.S)


# ---------------------------------------------------------------- model
def _download_model() -> Path:
    from huggingface_hub import hf_hub_download
    out = HERE / "model_cache"
    for name in ("model.safetensors", "config.json", "tokenizer.json",
                 "tokenizer_config.json"):
        hf_hub_download(repo_id=HF_REPO, filename=f"{HF_SUBDIR}/{name}",
                        local_dir=out)
    return out / HF_SUBDIR


def load_model() -> None:
    global MODEL, TOK, CHAT_TEMPLATE
    try:
        src = Path(MODEL_DIR) if MODEL_DIR else _download_model()
        tok = Tokenizer.from_file(str(src / "tokenizer.json"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        cfg = json.loads((src / "tokenizer_config.json").read_text())
        CHAT_TEMPLATE = cfg["chat_template"]
        model = Nae1ForCausalLM.from_pretrained(
            str(src), torch_dtype=DTYPE)
        model.eval()
        # The checkpoint stores no lm_head tensor (embeddings are tied).
        # Some transformers versions materialize weights in a way that
        # breaks the __init__ alias and never re-tie, leaving lm_head
        # uninitialized — generation then degenerates to a single token
        # (observed in production: endless <|pad|>). Re-tie explicitly in
        # plain PyTorch so the load is correct on any framework version.
        if getattr(model.config, "tie_word_embeddings", False):
            model.lm_head.weight = model.model.embed_tokens.weight
        model.to(DEVICE)
        TOK, MODEL = tok, model
        STATE["ready"] = True
        with torch.no_grad():
            probe = generate("The capital of France is", 6, 0.0, 1.0,
                             stop_on_call=False)
        print(f"[ready] {MODEL_LABEL} loaded from {src} | probe: {probe!r}",
              flush=True)
    except Exception as exc:  # keep /health answering so the failure is visible
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[fatal] model load failed: {STATE['error']}", flush=True)


# ---------------------------------------------------------------- tokens
def count_tokens(text: str) -> int:
    return len(TOK.encode(text).ids)


def truncate_to_budget(text: str, budget: int = MAX_TOOL_TOKENS) -> str:
    ids = TOK.encode(text).ids
    if len(ids) <= budget:
        return text
    keep = budget - count_tokens(" […]")
    return TOK.decode(ids[:keep]) + " […]"


def tool_block(name: str, arg: str, body: str) -> str:
    return f'<<tool name="{name}" query="{arg}">>{body}<</tool>>'


def error_block(name: str, arg: str, code: str, msg: str) -> str:
    return f'<<tool name="{name}" query="{arg}">><error>{code}: {msg}</error><</tool>>'


def find_complete_call(text: str, *, eof: bool):
    if not eof:
        nl = text.rfind("\n")
        if nl < 0:
            return None
        window = text[: nl + 1]
    else:
        window = text
    m = _CALL_RE.search(window)
    if not m:
        return None
    name = m.group("name")
    if name not in TOOLS:
        return None
    return name, m.group("arg")[1:-1]


# ---------------------------------------------------------------- tools
def _public_url(url: str) -> bool:
    try:
        host = urllib.request.urlparse(url).hostname or ""
        if not host or host == "localhost":
            return False
        infos = socket.getaddrinfo(host, None)
        return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)
    except Exception:
        return False


def exec_tool(name: str, arg: str) -> str:
    if name == "search":
        return error_block(name, arg, "search_unavailable",
                           "no search backend is configured on this deployment")
    if name == "fetch":
        if not arg.startswith(("https://", "http://")) or not _public_url(arg):
            return error_block(name, arg, "blocked_url",
                               "only public http(s) URLs can be fetched")
        try:
            req = urllib.request.Request(arg, headers={"User-Agent": "NeuralAI-Nae1/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = resp.read(65536).decode("utf-8", "replace")
            text = re.sub(r"<script.*?</script>|<style.*?</style>", " ",
                          raw, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return tool_block(name, arg, truncate_to_budget(text))
        except Exception as exc:
            return error_block(name, arg, "fetch_failed", str(exc)[:160])
    if name == "read_file":
        try:
            path = (DATA_DIR / arg).resolve()
            if DATA_DIR.resolve() not in path.parents and path != DATA_DIR.resolve():
                return error_block(name, arg, "blocked_path",
                                   "files are limited to the data directory")
            if not path.is_file():
                return error_block(name, arg, "not_found",
                                   "no such file in the data directory")
            return tool_block(name, arg,
                              truncate_to_budget(path.read_text("utf-8", "replace")))
        except Exception as exc:
            return error_block(name, arg, "read_failed", str(exc)[:160])
    return error_block(name, arg, "unknown_tool", name)


# ---------------------------------------------------------------- generation
class ClientGone(Exception):
    """The streaming client disconnected; abort generation and free the lock."""


def _pick(logits: torch.Tensor, out: list[int], temperature: float,
          top_p: float, rep_penalty: float, min_p: float = 0.0) -> int:
    if rep_penalty and rep_penalty != 1.0 and out:
        seen = torch.tensor(sorted(set(out)), dtype=torch.long,
                            device=logits.device)
        s = logits[seen]
        logits[seen] = torch.where(s < 0, s * rep_penalty, s / rep_penalty)
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        if min_p and min_p > 0:
            # min-p: drop tokens below min_p x the top token's probability.
            # Trims the garbage tail that dominates at high temperature on a
            # small model, without touching the plausible head.
            probs = torch.where(probs >= min_p * probs.max(), probs,
                                torch.zeros_like(probs))
        vals, idx = torch.sort(probs, descending=True)
        cume = torch.cumsum(vals, dim=0)
        keep = cume - vals < top_p
        keep[0] = True
        return int(idx[torch.multinomial(vals * keep, 1)].item())
    return int(torch.argmax(logits).item())


@torch.no_grad()
def generate(prompt: str, max_new: int, temperature: float, top_p: float,
             stop_on_call: bool, rep_penalty: float = 1.0,
             on_token=None, min_p: float = 0.0) -> str:
    """KV-cached generation. The naive full-forward-per-token version took
    >10s/token on the free-tier CPU share; with the cache each step is a
    single-token forward. Sampling math is unchanged (verified token-for-
    token against the uncached path)."""
    ids = TOK.encode(prompt).ids[-(N_CTX - CTX_RESERVE):]
    dev = next(MODEL.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    out: list[int] = []
    res = MODEL(input_ids=x, use_cache=True)
    logits = res["logits"][0, -1].float()
    past = res["past_key_values"]
    for _ in range(max_new):
        nxt = _pick(logits, out, temperature, top_p, rep_penalty, min_p)
        if nxt == EOS_ID:
            break
        out.append(nxt)
        if on_token:
            on_token(TOK.decode(out))
        if stop_on_call and len(out) % 4 == 0:
            if find_complete_call(TOK.decode(out), eof=False):
                break
        res = MODEL(input_ids=torch.tensor([[nxt]], device=dev),
                    past_key_values=past, use_cache=True)
        logits = res["logits"][0, -1].float()
        past = res["past_key_values"]
    return TOK.decode(out)


def render_prompt(messages: list[dict]) -> str:
    import jinja2
    return jinja2.Template(CHAT_TEMPLATE).render(
        messages=messages, add_generation_prompt=True)


def visible_text(model_text: str) -> str:
    out = _BLOCK_RE.sub("", model_text)
    return out.replace("»", "🔧").strip()


def chat(messages: list[dict], max_new: int, temperature: float,
         top_p: float, rep_penalty: float = 1.0, emit=None,
         min_p: float = 0.0) -> dict:
    prompt = render_prompt(messages)
    transcript = prompt
    calls = []
    reply = ""
    prior_raw = ""
    for _ in range(MAX_ROUNDS + 1):
        cb = None
        if emit:
            def cb(round_text, _prior=prior_raw):
                emit(visible_text(_prior + round_text))
        with GEN_LOCK:
            gen = generate(transcript, max_new, temperature, top_p,
                           stop_on_call=True, rep_penalty=rep_penalty,
                           on_token=cb, min_p=min_p)
        call = find_complete_call(gen, eof=True)
        if not call:
            reply = gen
            transcript += gen
            break
        name, arg = call
        # Byte-true continuation (mirrors the training traces and the
        # golden40 phase-B prompt): keep the generated bytes through the
        # end of the call line — including the leading space that the
        # fused token 4776 carries — then "\n", the result block, "\n",
        # and the SAME assistant turn continues. No second "NeuralAI:".
        m = _CALL_RE.search(gen)
        block = exec_tool(name, arg)
        transcript += gen[: m.end()] + "\n" + block + "\n"
        prior_raw += gen[: m.end()] + "\n" + block + "\n"
        calls.append({"tool": name, "arg": arg})
        reply = gen[: m.end()]
    return {"reply": visible_text(reply), "tool_calls": calls,
            "transcript": transcript}


# ---------------------------------------------------------------- http
def _bearer(handler) -> str:
    auth = handler.headers.get("Authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else ""


def _chat_auth_ok(handler) -> bool:
    """The public site's worker attaches the token as X-Nae1-Auth (its
    long-standing proxy convention); direct API clients use Bearer."""
    if not CHAT_TOKEN:
        return False
    if hmac.compare_digest(_bearer(handler), CHAT_TOKEN):
        return True
    return hmac.compare_digest(handler.headers.get("X-Nae1-Auth", ""), CHAT_TOKEN)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def setup(self):
        super().setup()
        # No handler may wedge a connection forever (a stalled socket once
        # held the generation lock hostage through a dead stream).
        self.connection.settimeout(180)

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/health":
            if not STATE["ready"]:
                return self._json(200, {"status": "loading" if not STATE["error"] else "error",
                                        "model": MODEL_LABEL, "detail": STATE["error"]})
            return self._json(200, {
                "status": "ok", "model": MODEL_LABEL, "model_type": "nae1",
                "tokenizer_vocab": 32000, "context": N_CTX,
                "params": 100111872, "step": STEP, "heldout_ppl": GUARDRAIL_PPL,
                "engine": f"torch (native Nae1ForCausalLM, {_DTYPE_NAME})",
                "device": DEVICE})
        if self.path == "/diag":
            if not DIAG_KEY or not hmac.compare_digest(_bearer(self), DIAG_KEY):
                return self._json(401, {"error": "unauthorized"})
            return self._json(200, {"ready": STATE["ready"], "error": STATE["error"],
                                    "uptime_s": round(time.time() - STATE["started"]),
                                    "model": MODEL_LABEL, "hf_source": f"{HF_REPO}/{HF_SUBDIR}"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/chat", "/generate", "/abort"):
            return self._json(404, {"error": "not found"})
        if not _chat_auth_ok(self):
            return self._json(401, {"error": "unauthorized"})
        if self.path == "/abort":
            # API parity with the site worker: generations are short and
            # synchronous here, so there is never one in flight to stop.
            return self._json(200, {"ok": True, "aborted": 0})
        if not STATE["ready"]:
            return self._json(503, {"error": "model not ready", "detail": STATE["error"]})
        try:
            req = self._body()
        except Exception:
            return self._json(400, {"error": "invalid json"})
        max_new = min(int(req.get("max_tokens", req.get("max_new_tokens", 160))), 512)
        temperature = float(req.get("temperature", 0.7))
        top_p = float(req.get("top_p", 0.9))
        rep = float(req.get("repetition_penalty", 1.0))
        min_p = min(max(float(req.get("min_p", 0.0)), 0.0), 0.5)
        if self.path == "/generate":
            with GEN_LOCK:
                text = generate(str(req.get("prompt", "")), max_new,
                                temperature, top_p, stop_on_call=False,
                                rep_penalty=rep, min_p=min_p)
            return self._json(200, {"text": text})
        if "prompt" in req or "messages" in req:
            return self._site_stream(req, max_new, temperature, top_p, rep, min_p)
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._json(400, {"error": "messages required"})
        out = chat(messages, max_new, temperature, top_p, rep_penalty=rep, min_p=min_p)
        return self._json(200, {"reply": out["reply"], "tool_calls": out["tool_calls"]})

    def _site_stream(self, req, max_new, temperature, top_p, rep, min_p=0.0):
        """Site protocol (the public chat page): body {prompt, temperature,
        top_k, top_p, max_new_tokens, repetition_penalty} — or, since the
        2026-10-21 site rebuild, {messages: [{role, content}, ...]} for a
        multi-turn conversation, rendered through the same chat template;
        response is an SSE-style stream of `data: {"text": <cumulative
        visible text>, "done": bool}` lines, ending with done=true +
        eos=true."""
        self.connection.settimeout(30)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        state = {"last": "", "dead": False}

        def frame(obj):
            if state["dead"]:
                return
            try:
                self.wfile.write(("data: " + json.dumps(obj) + "\n").encode())
                self.wfile.flush()
            except OSError:
                state["dead"] = True

        def emit(text):
            if state["dead"]:
                raise ClientGone()
            state["last"] = text
            frame({"text": text, "done": False})
            if state["dead"]:
                raise ClientGone()

        try:
            self.wfile.write(b": neuralai stream open\n")
            self.wfile.flush()
        except OSError:
            return
        try:
            raw_msgs = req.get("messages")
            if isinstance(raw_msgs, list) and raw_msgs:
                convo = []
                for m in raw_msgs[-24:]:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role", "user"))
                    if role not in ("user", "assistant", "system"):
                        role = "user"
                    convo.append({"role": role,
                                  "content": str(m.get("content", ""))})
                if not convo:
                    convo = [{"role": "user",
                              "content": str(req.get("prompt", ""))}]
            else:
                convo = [{"role": "user",
                          "content": str(req.get("prompt", ""))}]
            out = chat(convo,
                       max_new, temperature, top_p, rep_penalty=rep, emit=emit,
                       min_p=min_p)
            frame({"text": out["reply"], "done": True, "eos": True})
        except ClientGone:
            return
        except Exception as exc:
            print(f"[site-stream] error: {exc}", flush=True)
            frame({"text": state["last"], "done": True, "eos": False})


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("BLOCKING_LOAD"):
        # Modal: bind the port only once the model is ready, so the
        # platform's startup wait covers the full cold start and no
        # request can land on a not-ready server.
        load_model()
    else:
        threading.Thread(target=load_model, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[listen] :{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
