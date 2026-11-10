#!/usr/bin/env python3
"""NeuralAI — Nae1 inference service (rebuilt 2026-10-21 for the SFT v8 release).

Serves the SFT v13.2 LoRA assistant checkpoint (step-12000, adapters merged)
natively in PyTorch bf16
on CPU. Endpoints:

  GET  /health    public; deployment health + model facts (shape kept from
                  the original service)
  POST /chat      Bearer CHAT_TOKEN; {messages:[{role,content}], ...}
                  runs the trained tool loop (search/fetch/read_file/calc)
  POST /generate  Bearer CHAT_TOKEN; {prompt, ...} raw completion
  GET  /diag      Bearer DIAG_KEY; runtime diagnostics
  GET  /v1/models
  GET  /v1/models/<id>
  POST /v1/chat/completions
                  Bearer CHAT_TOKEN (or X-Nae1-Auth); OpenAI-compatible,
                  non-streaming. Browser-safe: answers CORS preflights and
                  sends Access-Control-Allow-Origin on every /v1 response,
                  so a web page (e.g. OpenStudio) can call it directly.

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

import ast
import hmac
import ipaddress
import json
import os
import re
import socket
import threading
import time
import urllib.request
from urllib.parse import unquote, urlsplit
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
HF_SUBDIR = os.environ.get("HF_SUBDIR", "sft_v13_2_lora_step12000_bf16")
DATA_DIR = Path(os.environ.get("DATA_DIR", str(HERE / "data")))
EOS_ID = 3
N_CTX = 2048
CTX_RESERVE = 256
MAX_ROUNDS = 3
MAX_TOOL_TOKENS = 250
TOOLS = ("search", "fetch", "read_file", "calc")
MODEL_LABEL = "nae1-sft-v13-2-lora-step12000 (bf16)"
# Clean, typable id for the OpenAI-compatible API (the label above carries
# spaces and parens, which clients would have to URL-encode everywhere).
API_MODEL_ID = "nea1-v13.2"
STEP = 12000
# 50-doc guardrail set (fluency_docs_20261017), n_ctx 512 — the only PPL
# measured for this checkpoint; see the model card for the protocol note.
GUARDRAIL_PPL = 48.01
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


def _calc_eval(expr: str):
    """Safe arithmetic only: numbers and + - * / % ** ( ). No names,
    calls, or attribute access, so an expression can never run code."""
    if len(expr) > 200:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                    node.value, (int, float)):
                raise ValueError("not a number")
            return node.value
        if isinstance(node, ast.BinOp):
            left = ev(node.left)
            right = ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                if abs(right) > 100 or abs(left) > 1000000:
                    raise ValueError("power too large")
                return left ** right
            raise ValueError("operator not allowed")
        if isinstance(node, ast.UnaryOp):
            value = ev(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.USub):
                return -value
            raise ValueError("operator not allowed")
        raise ValueError("not arithmetic")

    return ev(tree)


def _calc_format(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


# ------------------------------------------------------- deterministic calc
# STAGED 2026-10-30 (Dre's go: "confirm build it" — deterministic calc
# layer, NOT YET DEPLOYED; deploy only with Dre's separate deploy go).
# Why: five SFT builds (v13.3–v13.7) never got the model to copy digits
# reliably (best 29/40, stubborn no-calls + attractor drifts). The
# calculator itself was never wrong — only the model's copying was. So
# direct arithmetic questions bypass model generation entirely: the
# server extracts the expression from the question, evaluates it with
# the same safe evaluator the calc tool uses, and answers
# deterministically. The model path is untouched for everything else
# (open-ended chat, multi-step reasoning, non-direct questions), and
# any detection/evaluation failure falls through to the model path —
# fail-open, never fail-wrong.
_DIRECT_ARITH_RE = re.compile(
    r"^\s*(?:what\s+is|what's|calculate|work\s+out|compute)\s+"
    r"([\d\s\.\+\-\*/%\(\)]+?)\s*\??\s*$", re.IGNORECASE)
_EXPR_CHARS_RE = re.compile(r"^[\d\s\.\+\-\*/%\(\)]+$")


def detect_direct_arithmetic(text: str):
    """Return the arithmetic expression if `text` is a single direct
    arithmetic question, else None. Strict by design: the whole message
    must be one opener + one pure expression — no prose, no second
    clause, no words inside the expression. Anything else -> None and
    the caller uses the model path."""
    if not isinstance(text, str):
        return None
    m = _DIRECT_ARITH_RE.match(text.strip())
    if not m:
        return None
    expr = re.sub(r"\s+", " ", m.group(1)).strip()
    # A trailing sentence period is punctuation, not part of the number
    # ("Calculate 1000 / 8." -> "1000 / 8"). A decimal point inside the
    # expression is untouched (only one trailing dot is stripped).
    if expr.endswith("."):
        expr = expr[:-1].rstrip()
    if not expr or not _EXPR_CHARS_RE.match(expr):
        return None
    if expr.count("(") != expr.count(")"):
        return None
    # A bare number is not a question ("What is 42?" -> no operator).
    if not re.search(r"[\+\-\*/%]", expr):
        return None
    return expr


def exec_tool(name: str, arg: str):
    """Execute one tool call. Returns (block, outcome) where block is the
    byte-true result block for the model transcript (unchanged by this
    function — the transcript assembly must match training traces
    exactly) and outcome is a small display dict for the tool-call UI:
    {"ok": bool, "status": short outcome string, "detail": error
    message or None}. Outcome text is never fed to the model."""
    if name == "calc":
        # STAGED 2026-10-30 (v13.6 format fix, NOT YET DEPLOYED — deploy
        # only with Dre's promotion go): v13.6 call lines carry a leading
        # space inside the quotes (» calc(" 126 + 359")) so call tokens
        # match question tokens exactly. ast.parse rejects leading
        # whitespace, so evaluate the stripped query; the block still
        # echoes the original arg so inference context matches training
        # traces byte-for-byte. Stripping is a no-op for all old-format
        # queries, so this changes nothing for the currently serving model.
        try:
            value = _calc_eval(arg.strip())
        except ZeroDivisionError:
            return error_block(name, arg, "math_error", "division by zero"), \
                {"ok": False, "status": "math_error",
                 "detail": "division by zero"}
        except Exception:
            return error_block(name, arg, "invalid_expression",
                               "only numbers and + - * / % ** ( ) are allowed"), \
                {"ok": False, "status": "invalid_expression",
                 "detail": "only numbers and + - * / % ** ( ) are allowed"}
        return tool_block(name, arg, _calc_format(value)), \
            {"ok": True, "status": "ok", "detail": None}
    if name == "search":
        return error_block(name, arg, "search_unavailable",
                           "no search backend is configured on this deployment"), \
            {"ok": False, "status": "unavailable",
             "detail": "no search backend is configured on this deployment"}
    if name == "fetch":
        if not arg.startswith(("https://", "http://")) or not _public_url(arg):
            return error_block(name, arg, "blocked_url",
                               "only public http(s) URLs can be fetched"), \
                {"ok": False, "status": "blocked_url",
                 "detail": "only public http(s) URLs can be fetched"}
        try:
            req = urllib.request.Request(arg, headers={"User-Agent": "NeuralAI-Nae1/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                status = resp.status
                raw = resp.read(65536).decode("utf-8", "replace")
            text = re.sub(r"<script.*?</script>|<style.*?</style>", " ",
                          raw, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return tool_block(name, arg, truncate_to_budget(text)), \
                {"ok": True, "status": f"{status} OK", "detail": None}
        except Exception as exc:
            return error_block(name, arg, "fetch_failed", str(exc)[:160]), \
                {"ok": False, "status": "fetch_failed",
                 "detail": str(exc)[:160]}
    if name == "read_file":
        try:
            path = (DATA_DIR / arg).resolve()
            if DATA_DIR.resolve() not in path.parents and path != DATA_DIR.resolve():
                return error_block(name, arg, "blocked_path",
                                   "files are limited to the data directory"), \
                    {"ok": False, "status": "blocked_path",
                     "detail": "files are limited to the data directory"}
            if not path.is_file():
                return error_block(name, arg, "not_found",
                                   "no such file in the data directory"), \
                    {"ok": False, "status": "not_found",
                     "detail": "no such file in the data directory"}
            return tool_block(name, arg,
                              truncate_to_budget(path.read_text("utf-8", "replace"))), \
                {"ok": True, "status": "ok", "detail": None}
        except Exception as exc:
            return error_block(name, arg, "read_failed", str(exc)[:160]), \
                {"ok": False, "status": "read_failed",
                 "detail": str(exc)[:160]}
    return error_block(name, arg, "unknown_tool", name), \
        {"ok": False, "status": "unknown_tool", "detail": name}


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
         min_p: float = 0.0, on_tool=None) -> dict:
    # on_tool, when given, is called with one display event per tool
    # fire ({"tool", "arg", "ok", "status", "detail"?}) as the tool
    # executes. The event is a display-only channel: it never enters
    # the transcript and never changes what the model sees. The
    # transcript assembly below is byte-true to the training traces.
    # Deterministic calc layer (staged 2026-10-30): a direct arithmetic
    # question as the last user message is answered without generation.
    # Any detection or evaluation failure falls through to the model.
    if messages and messages[-1].get("role") == "user":
        expr = detect_direct_arithmetic(messages[-1].get("content", ""))
        if expr is not None:
            try:
                reply = "The answer is " + _calc_format(_calc_eval(expr)) + "."
            except Exception:
                reply = None
            if reply is not None:
                prompt = render_prompt(messages)
                return {"reply": reply,
                        "tool_calls": [{"tool": "calc", "arg": expr,
                                        "deterministic": True}],
                        "transcript": prompt + reply}
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
        block, outcome = exec_tool(name, arg)
        transcript += gen[: m.end()] + "\n" + block + "\n"
        prior_raw += gen[: m.end()] + "\n" + block + "\n"
        ev = {"tool": name, "arg": arg, "ok": outcome["ok"],
              "status": outcome["status"]}
        if outcome.get("detail"):
            ev["detail"] = outcome["detail"]
        calls.append(ev)
        if on_tool:
            on_tool(ev)
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

    def _json(self, code: int, obj: dict, cors: bool = False):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                "authorization, content-type, x-nae1-auth")
        self.end_headers()
        self.wfile.write(body)

    def _v1_model_object(self) -> dict:
        return {"id": API_MODEL_ID, "object": "model",
                "created": int(STATE["started"]), "owned_by": "neuralai"}

    def _v1_error(self, code: int, message: str, err_code: str):
        return self._json(code, {"error": {
            "message": message, "type": "invalid_request_error",
            "code": err_code}}, cors=True)

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
        v1path = urlsplit(self.path).path
        if v1path == "/v1/models":
            if not _chat_auth_ok(self):
                return self._json(401, {"error": "unauthorized"}, cors=True)
            return self._json(200, {"object": "list",
                                    "data": [self._v1_model_object()]}, cors=True)
        if v1path.startswith("/v1/models/"):
            if not _chat_auth_ok(self):
                return self._json(401, {"error": "unauthorized"}, cors=True)
            wanted = unquote(v1path[len("/v1/models/"):])
            if wanted in (API_MODEL_ID, MODEL_LABEL):
                return self._json(200, self._v1_model_object(), cors=True)
            return self._v1_error(404, f"model not found: {wanted}",
                                  "model_not_found")
        return self._json(404, {"error": "not found"},
                          cors=v1path.startswith("/v1/"))

    def do_OPTIONS(self):
        # Browser preflight for the /v1 API (a web page calling the model
        # directly). Other paths keep the default 501 behavior untouched.
        if urlsplit(self.path).path.startswith("/v1/"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                "authorization, content-type, x-nae1-auth")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        v1path = urlsplit(self.path).path
        is_v1 = v1path == "/v1/chat/completions"
        if self.path not in ("/chat", "/generate", "/abort") and not is_v1:
            return self._json(404, {"error": "not found"},
                              cors=v1path.startswith("/v1/"))
        if not _chat_auth_ok(self):
            return self._json(401, {"error": "unauthorized"}, cors=is_v1)
        if self.path == "/abort":
            # API parity with the site worker: generations are short and
            # synchronous here, so there is never one in flight to stop.
            return self._json(200, {"ok": True, "aborted": 0})
        if not STATE["ready"]:
            return self._json(503, {"error": "model not ready",
                                    "detail": STATE["error"]}, cors=is_v1)
        try:
            req = self._body()
        except Exception:
            return self._json(400, {"error": "invalid json"}, cors=is_v1)
        max_new = min(int(req.get("max_tokens", req.get("max_new_tokens", 160))), 512)
        temperature = float(req.get("temperature", 0.7))
        top_p = float(req.get("top_p", 0.9))
        rep = float(req.get("repetition_penalty", 1.0))
        min_p = min(max(float(req.get("min_p", 0.0)), 0.0), 0.5)
        if is_v1:
            return self._v1_chat_completions(req, max_new, temperature,
                                             top_p, rep, min_p)
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

    def _v1_chat_completions(self, req, max_new, temperature, top_p, rep, min_p):
        """OpenAI-compatible chat completion.

        Request: {model?, messages: [{role, content}], temperature?,
        top_p?, max_tokens?, repetition_penalty?, min_p?, stream?}.
        Unknown fields (e.g. response_format) are ignored. The served
        model is fixed, so a model name that is not ours is a 404, not
        a silent substitution. stream=true returns OpenAI-style SSE
        chunks (see _v1_stream); the reply is generated in full first,
        exactly as the non-streaming path does, then emitted as delta
        chunks — there is no token-incremental generation loop here.
        """
        wanted = req.get("model")
        if wanted is not None and str(wanted) not in (API_MODEL_ID, MODEL_LABEL):
            return self._v1_error(404, f"model not found: {wanted}",
                                  "model_not_found")
        raw_msgs = req.get("messages")
        if not isinstance(raw_msgs, list) or not raw_msgs:
            return self._v1_error(400, "messages required",
                                  "invalid_request_error")
        convo = []
        for m in raw_msgs[-24:]:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user"))
            if role not in ("user", "assistant", "system"):
                role = "user"
            convo.append({"role": role, "content": str(m.get("content", ""))})
        if not convo:
            return self._v1_error(400, "messages required",
                                  "invalid_request_error")
        try:
            out = chat(convo, max_new, temperature, top_p,
                       rep_penalty=rep, min_p=min_p)
        except Exception as exc:
            return self._v1_error(500, f"generation failed: {exc}",
                                  "internal_error")
        if req.get("stream"):
            return self._v1_stream(out["reply"], out["tool_calls"])
        reply = out["reply"]
        prompt_tokens = count_tokens(render_prompt(convo))
        completion_tokens = count_tokens(reply)
        return self._json(200, {
            "id": "chatcmpl-nea1-" + format(int(time.time()), "x"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": API_MODEL_ID,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": reply},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        }, cors=True)

    def _v1_stream(self, reply, tool_calls):
        """OpenAI-compatible SSE stream for /v1/chat/completions.

        Emits: a role chunk, the reply in a few word-boundary chunks
        (not one byte at a time, not one giant frame), a final chunk
        with delta {} and finish_reason "stop", then `data: [DONE]`.
        tool_calls ride the final chunk when the reply fired tools —
        same events the /chat path returns. A client that disconnects
        mid-stream just ends the write; nothing else is affected.
        """
        cid = "chatcmpl-nea1-" + format(int(time.time()), "x")
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

        def frame(obj):
            try:
                self.wfile.write(("data: " + json.dumps(obj) + "\n").encode())
                self.wfile.flush()
            except OSError:
                raise ClientGone()

        def chunk(delta, finish_reason):
            return {"id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": API_MODEL_ID,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish_reason}]}

        try:
            frame(chunk({"role": "assistant", "content": ""}, None))
            buf = ""
            for word in str(reply).split(" "):
                piece = word if not buf else " " + word
                if buf and len(buf) + len(piece) > 48:
                    frame(chunk({"content": buf}, None))
                    buf = word
                else:
                    buf += piece
            if buf:
                frame(chunk({"content": buf}, None))
            final = chunk({}, "stop")
            if tool_calls:
                final["choices"][0]["tool_calls"] = tool_calls
            frame(final)
            self.wfile.write(b"data: [DONE]\n")
            self.wfile.flush()
        except (OSError, ClientGone):
            return

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

        def on_tool(ev):
            # Display-only tool event: one frame per tool fire, carrying
            # the tool name, arg, and outcome. The page renders it as a
            # collapsible details block. Never enters model context.
            frame({"tool": ev})

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
                       on_tool=on_tool, min_p=min_p)
            # tool_calls (with outcomes) rides the final frame so the page
            # can render details blocks even if a mid-stream tool frame
            # was dropped; the page skips entries it already rendered.
            frame({"text": out["reply"], "done": True, "eos": True,
                   "tool_calls": out["tool_calls"]})
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
