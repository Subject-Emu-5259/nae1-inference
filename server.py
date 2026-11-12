#!/usr/bin/env python3
"""NeuralAI — Nae1 inference service (rebuilt 2026-10-21 for the SFT v8 release).

Serves the SFT v17 LoRA assistant checkpoint (step-9250, adapters merged)
natively in PyTorch bf16
on CPU. Endpoints:

  GET  /health    public; deployment health + model facts (shape kept from
                  the original service)
  POST /chat      Bearer CHAT_TOKEN; {messages:[{role,content}], ...}
                  runs the trained tool loop (fetch/calc — search and
                  read_file were removed from the callable set
                  2026-11-12: no search backend exists and the data
                  directory is empty, so both tools could only fail;
                  their exec_tool handlers remain for the gate harness,
                  which calls them directly)
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
import datetime
import hmac
import ipaddress
import json
import math
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
HF_SUBDIR = os.environ.get("HF_SUBDIR", "sft_v17_lora_step9250_bf16")
DATA_DIR = Path(os.environ.get("DATA_DIR", str(HERE / "data")))
EOS_ID = 3
N_CTX = 2048
CTX_RESERVE = 256
MAX_ROUNDS = 3
MAX_TOOL_TOKENS = 250
MAX_FETCH_TOKENS = 500  # SVC-P2-11: fetch gets a bigger window than the
# default tool budget — a head-only 250-token cut hides answers that sit
# later in the page. Still bounded: one fetch block is a quarter of ctx.
# SVC-P0-2 (2026-11-12): search and read_file are NOT callable — no
# search backend exists (search returned search_unavailable on every
# node) and the data directory is empty (read_file always not_found).
# Their exec_tool handlers below stay in place: the livetools gate
# harness calls them directly, and removing the handlers would break
# it. The model is offered only fetch + calc.
TOOLS = ("fetch", "calc")
MODEL_LABEL = "nae1-sft-v17-lora-step9250 (bf16)"
# Clean, typable id for the OpenAI-compatible API (the label above carries
# spaces and parens, which clients would have to URL-encode everywhere).
API_MODEL_ID = "nea1-v13.2"
STEP = 9250
# 50-doc guardrail set (fluency_docs_20261017), n_ctx 512 — the only PPL
# measured for this checkpoint; see the model card for the protocol note.
GUARDRAIL_PPL = 50.49
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


def _log(msg: str) -> None:
    """Timestamped stdout log (C-P1-1, 2026-11-12): the node logs had no
    timestamps at all, so 242 silent server deaths could not be dated
    or correlated with anything. Every server print goes through here."""
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}",
          flush=True)


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
        _log(f"[ready] {MODEL_LABEL} loaded from {src} | probe: {probe!r}")
    except Exception as exc:  # keep /health answering so the failure is visible
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        _log(f"[fatal] model load failed: {STATE['error']}")


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
# SVC-P0-2 (2026-11-12): the sandbox resolver maps every public hostname
# into 198.18.0.0/15 (CGNAT space, RFC 6598) — the egress proxy's NAT
# mapping for public hosts, not a private target. The old check required
# every resolved IP to be globally routable, so on the VM node every
# fetch died as blocked_url. A 198.18.x.x answer is public-equivalent
# HERE: CGNAT addresses are not routable on the public internet, so on
# Render/Mac (real DNS) a hostname genuinely resolving into 198.18/15
# simply fails to connect — the allowlist cannot smuggle a fetch to an
# internal service. Private ranges (10/8, 172.16/12, 192.168/16,
# 127/8, 169.254/16, ::1) are still rejected.
_SANDBOX_RESOLVER_NET = ipaddress.ip_network("198.18.0.0/15")


def _public_url(url: str) -> bool:
    try:
        parts = urllib.request.urlparse(url)
        # SVC-P1-3: http(s) only — the fetch call-site checks the scheme
        # too, but _public_url is also the redirect re-validator, and an
        # http->ftp redirect would otherwise re-validate as "public"
        # (the host resolves) and be fetched over FTP.
        if parts.scheme not in ("http", "https"):
            return False
        host = parts.hostname or ""
        if not host or host == "localhost":
            return False
        infos = socket.getaddrinfo(host, None)
        return all(ipaddress.ip_address(i[4][0]).is_global
                   or ipaddress.ip_address(i[4][0]) in _SANDBOX_RESOLVER_NET
                   for i in infos)
    except Exception:
        return False


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """SSRF guard (SVC-P1-3): urlopen follows redirects by default with
    no re-validation, so a public URL could 301 to 169.254.169.254 or
    127.0.0.1 and be fetched unchecked. Re-validate _public_url on
    every hop; a non-public redirect target aborts the fetch. Max 5
    hops (urllib's default is 10)."""
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _public_url(newurl):
            raise ValueError("redirect to non-public URL blocked")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Install the validating redirect handler as the process-wide opener so
# the fetch call-site keeps using plain urllib.request.urlopen — the
# livetools gate harness monkeypatches urlopen, and an opener.open()
# call-site would bypass its fixture redirect. No other code in this
# process uses urlopen.
urllib.request.install_opener(
    urllib.request.build_opener(_SafeRedirectHandler))


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
# EXTENDED 2026-11-12 (§8 coverage gap, parent-found by probing): the
# opener-only detector missed bare expressions ("8473 * 96") and word
# operators ("8473 times 96") — both fell through to the model, which
# answers wrong with confidence (verified: "8473 - 96 = 120"). The
# detector now also accepts a bare expression (no opener) and word
# operators (times/plus/minus/divided by/multiplied by), plus digit
# commas ("1,000 + 2"), ^/×/÷ symbols, and word numbers ("two plus
# two"). Normalization turns every accepted form into the same pure
# symbol expression the old regex validated, so the strictness
# guarantee (the whole message must be one expression) is unchanged —
# and anything unparseable still falls through to the model, fail-open.
_DIRECT_ARITH_RE = re.compile(
    r"^\s*(?:what\s+is|what's|calculate|work\s+out|compute)\s+"
    r"(.+?)\s*\??\s*$", re.IGNORECASE)
_EXPR_CHARS_RE = re.compile(r"^[\d\s\.\+\-\*/%\(\)]+$")
# Bare date shapes must NOT be read as arithmetic: a bare "2026-11-12"
# is a date the user typed, not 2026 minus 11 minus 12. (With an opener
# — "What is 2026-11-12?" — the question is unambiguous, so the guard
# applies to the bare form only.)
_BARE_DATE_RE = re.compile(
    r"^(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}-\d{1,2}-\d{4}|\d{1,2}/\d{1,2}/\d{2,4})$")
_WORD_OPS = [
    (re.compile(r"\bdivided\s+by\b", re.IGNORECASE), "/"),
    (re.compile(r"\bmultiplied\s+by\b", re.IGNORECASE), "*"),
    (re.compile(r"\btimes\b", re.IGNORECASE), "*"),
    (re.compile(r"\bplus\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE), "-"),
]
_NUM_WORD_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_NUM_WORD_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUM_WORDS = set(_NUM_WORD_ONES) | set(_NUM_WORD_TENS) | {"hundred", "thousand"}


def _number_words_value(words):
    """'two hundred fifty' -> 250; 'one thousand' -> 1000; a bare
    'hundred' reads as 100 ('a hundred' has no number word before it)."""
    total = 0
    current = 0
    for w in words:
        if w in _NUM_WORD_ONES:
            current += _NUM_WORD_ONES[w]
        elif w in _NUM_WORD_TENS:
            current += _NUM_WORD_TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w == "thousand":
            total += (current or 1) * 1000
            current = 0
    return total + current


def _words_to_digits(text: str) -> str:
    """Replace maximal runs of number words with digits ('two hundred
    fifty' -> '250'). 'and' inside a run is skipped ('one hundred and
    five' -> '105'); anything else ends the run."""
    out = []
    run = []

    def flush():
        if run:
            out.append(str(_number_words_value(run)))
            run.clear()

    for tok in text.split():
        low = tok.lower()
        if low in _NUM_WORDS:
            run.append(low)
        elif low == "and" and run:
            continue  # "one hundred and five": "and" is part of the number
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


def detect_direct_arithmetic(text: str):
    """Return the arithmetic expression if `text` is a single direct
    arithmetic question, else None. Strict by design: the whole message
    must be one opener + one expression, or one bare expression — no
    prose, no second clause. Word operators, word numbers, commas, and
    ^/×/÷ are normalized to pure symbols first; anything left over
    fails validation and the caller uses the model path."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    m = _DIRECT_ARITH_RE.match(t)
    bare = m is None
    expr_raw = t if bare else m.group(1)
    if bare and expr_raw.endswith("?"):
        expr_raw = expr_raw[:-1].rstrip()
    # Bare date shapes are not arithmetic (see _BARE_DATE_RE).
    if bare and _BARE_DATE_RE.match(expr_raw.strip()):
        return None
    # Normalize to pure symbols: ×/÷/^ first, then word operators,
    # then "negative N", then number words, then digit commas.
    expr = (expr_raw.replace("×", "*").replace("÷", "/")
            .replace("^", "**"))
    for pat, sym in _WORD_OPS:
        expr = pat.sub(sym, expr)
    # "negative five" -> "- 5": keep the space so the "-" doesn't fuse
    # with the number word ("-five" would fail validation below).
    expr = re.sub(r"\bnegative\s+", "- ", expr, flags=re.IGNORECASE)
    expr = _words_to_digits(expr)
    expr = re.sub(r"(?<=\d),(?=\d)", "", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    # A trailing sentence period is punctuation, not part of the number
    # ("Calculate 1000 / 8." -> "1000 / 8"). A decimal point inside the
    # expression is untouched (only one trailing dot is stripped).
    if expr.endswith("."):
        expr = expr[:-1].rstrip()
    if not expr or not _EXPR_CHARS_RE.match(expr):
        return None
    if expr.count("(") != expr.count(")"):
        return None
    # A bare number is not a question ("What is 42?" / "42" -> no operator).
    if not re.search(r"[\+\-\*/%]", expr):
        return None
    # A leading binary operator ("plus 5" -> "+ 5") is not a question —
    # only a leading "-" (a negative number) is legitimate there.
    if expr[0] in "+*/%":
        return None
    # Trial parse (no evaluation): catches trailing operators
    # ("2 + 2 plus?" -> "2 + 2 +") and other malformed shapes the
    # character check can't see. Anything unparseable falls through to
    # the model, fail-open.
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    return expr


# ------------------------------------------------------- deterministic date
# EXTENDED 2026-11-11 (Dre's go: "Extend the §8 date layer") — the same
# pattern as the §8 calc layer above: direct date questions bypass model
# generation entirely. v17 + three SFT patches all NO-GO'd on the date
# rollover, and v3 proved SFT teaches the format of a derivation, not
# the derivation (the model writes "+2" and computes +1; the year
# increment never generalizes). Python's datetime does the arithmetic;
# the layer only formats the answer. Fail-open: any detection or
# parsing failure falls through to the model path.
# "Today" is resolved bracket-first: the Workspace Mac app sends a
# system line "Today is Wednesday, November 11, 2026. ..." built from
# Dre's clock, and that always wins when present — it is the date the
# user sees. The public site chat sends no date at all (the Worker is
# a pure proxy), so the fallback is the server's local date, forced to
# America/Chicago below so every leg (Mac, VM, Modal, Render) agrees
# even though the VM/Modal/Render hosts run UTC. time.time() callers
# (uptime, completion ids) are TZ-independent, so forcing TZ here
# changes nothing else in this process.
# SVC-P1-5 (2026-11-12): this was os.environ.setdefault — a no-op on
# any host that already sets TZ (UTC is common on Render/VM images),
# which silently made "today" a day off for hours each night. Set it
# unconditionally now; main() logs the effective TZ at startup.
try:
    os.environ["TZ"] = "America/Chicago"
    time.tzset()
except Exception:
    pass

_MONTHS = {name: i for i, name in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
# Abbreviated months (SVC-P2-13, 2026-11-12): "Nov 11, 2026" in a
# Today line used to fail _DATE_RE entirely. "sept" is the common
# 4-letter September abbreviation; the regex allows an optional
# trailing period ("Nov. 11, 2026").
_MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
                "nov": 11, "dec": 12})
_MONTH_NAMES = {i: name for i, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")
_DATE_RE = re.compile(
    r"(?:(monday|tuesday|wednesday|thursday|friday|saturday|sunday),\s*)?"
    r"(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sept?\.?|oct\.?|nov\.?|dec\.?)\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.IGNORECASE)
_TODAY_LINE_RE = re.compile(r"today\s+is\s+", re.IGNORECASE)
_BRACKET_DATE_RE = re.compile(r"\[([^\]]+)\]")
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twenty one": 21, "twenty two": 22, "twenty three": 23,
    "twenty four": 24, "twenty five": 25, "twenty six": 26,
    "twenty seven": 27, "twenty eight": 28, "twenty nine": 29,
    "thirty": 30, "thirty one": 31,
}
_ONES = ["", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["", "", "twenty", "thirty"]


def _num_to_word(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n <= 31:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("-" + _ONES[ones] if ones else "")
    # Past thirty-one the word form is long ("forty-five"); digits read
    # fine in the answer ("In 45 days it will be ...").
    return str(n)


def _parse_date(text: str):
    """Parse 'Wednesday, November 11, 2026' (weekday optional) -> a
    datetime.date, else None. A present weekday that disagrees with
    the calendar rejects the line — never trust a wrong bracket."""
    if not isinstance(text, str):
        return None
    m = _DATE_RE.search(text)
    if not m:
        return None
    weekday, month, day, year = m.groups()
    try:
        d = datetime.date(int(year), _MONTHS[month.lower().rstrip(".")],
                          int(day))
    except ValueError:
        return None
    if weekday and d.strftime("%A").lower() != weekday.lower():
        return None
    return d


def _today_from_messages(messages):
    """'Today' from a caller-supplied date. Only system messages count,
    and only a 'Today is <date>' line or a '[<date>]' bracket — a user
    message mentioning a date ('my birthday is March 5, 2020') must
    never be read as today. The FIRST (newest) system message carrying
    a date decides: if it parses, use it; if it doesn't, fall back to
    the server's local date rather than trusting an older, staler one.
    Returns a datetime.date or None."""
    for m in reversed(messages or []):
        if not isinstance(m, dict) or m.get("role") != "system":
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or not content:
            continue
        tm = _TODAY_LINE_RE.search(content)
        if tm:
            d = _parse_date(content[tm.end():])
            return d  # None when unparseable: fall back, don't scan older
        for b in _BRACKET_DATE_RE.findall(content):
            d = _parse_date(b)
            if d is not None:
                return d
        # A system message with no date: keep scanning older ones.
    return None


def _fmt_date(d) -> str:
    return "%s, %s %d, %d" % (d.strftime("%A"), d.strftime("%B"),
                              d.day, d.year)


def _parse_num(token: str):
    """'two' / 'twenty one' / '14' / '45' -> int 1..365, else None.
    SVC-P2-13 (2026-11-12): the old 1..31 cap made 'in 45 days' fall
    through to the model; the horizon is now a year. Word numbers
    still top out at thirty-one (the _NUMBER_WORDS table); digits
    cover the rest."""
    token = (token or "").strip().lower()
    if token.isdigit():
        n = int(token)
        return n if 1 <= n <= 365 else None
    return _NUMBER_WORDS.get(token)


_NUM_ALT = (r"\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty(?: one| two| three| four| five| six|"
            r" seven| eight| nine)?|thirty(?: one)?")


def detect_direct_date(text: str, messages=None):
    """Return the one-sentence answer if `text` is a single direct date
    question, else None. Strict by design, mirroring the §8 calc
    detector: the whole message must be one question — no prose, no
    second clause. Anything ambiguous, multi-clause, or unparseable ->
    None and the caller uses the model path. A bare date with no
    question ('What is August 31, 2025?') is not a date question."""
    if not isinstance(text, str):
        return None
    # Normalize like the Workspace detectors: collapse whitespace,
    # lowercase, strip apostrophes, hyphens -> spaces ("twenty-one"
    # and "twenty one" behave the same).
    t = " ".join(text.split()).strip().lower().replace("'", "")
    t = t.replace("-", " ")
    if not t:
        return None

    today = _today_from_messages(messages)
    if today is None:
        today = datetime.date.today()

    # Weekday of an explicit date: "what day of the week is
    # September 1, 2025?" / "what day is November 11, 2026?"
    m = re.match(r"^what (?:day of the week|day|weekday) (?:is|was) (.+?)\s*[?.!]*$",
                 t)
    if m:
        d = _parse_date(m.group(1))
        if d is not None:
            return "%s %d, %d is a %s." % (d.strftime("%B"), d.day,
                                           d.year, d.strftime("%A"))
        # A date-shaped question that doesn't parse is an INVALID date
        # ("what day is February 30, 2025?") — say so deterministically
        # instead of letting the model hallucinate a weekday (SVC-P2-13).
        # Only when a month/day/year actually matched; "what day is it?"
        # has no date shape and falls through to the checks below.
        dm = _DATE_RE.search(m.group(1))
        if dm is not None:
            _wd, month, day, year = dm.groups()
            try:
                datetime.date(int(year),
                              _MONTHS[month.lower().rstrip(".")], int(day))
            except ValueError:
                return "%s %d, %d is not a valid calendar date." % (
                    _MONTH_NAMES[_MONTHS[month.lower().rstrip(".")]],
                    int(day), int(year))
            # The date exists but the stated weekday disagrees with the
            # calendar — fall through rather than calling a real date
            # invalid.
        # No parseable date ("what day is it?") — fall through to the
        # today/tomorrow checks below, never return None here.

    # "in N days": "what date is it in two days?" / "what day will it
    # be in two days?" / "what will the date be in 5 days?"
    m = re.match(r"^what (?:day|date)(?: is it| will it be)? in (" + _NUM_ALT +
                 r") days?\s*[?.!]*$", t)
    if not m:
        m = re.match(r"^what will the date be in (" + _NUM_ALT +
                     r") days?\s*[?.!]*$", t)
    if m:
        n = _parse_num(m.group(1))
        if n is None:
            return None
        return "In %s day%s it will be %s." % (
            _num_to_word(n), "" if n == 1 else "s",
            _fmt_date(today + datetime.timedelta(days=n)))

    # "N days ago": "what date was it two days ago?"
    m = re.match(r"^what (?:day|date)(?: was it| was)? (" + _NUM_ALT +
                 r") days? ago\s*[?.!]*$", t)
    if not m:
        m = re.match(r"^what was the date (" + _NUM_ALT +
                     r") days? ago\s*[?.!]*$", t)
    if m:
        n = _parse_num(m.group(1))
        if n is None:
            return None
        return "%s day%s ago it was %s." % (
            _num_to_word(n).capitalize(), "" if n == 1 else "s",
            _fmt_date(today - datetime.timedelta(days=n)))

    # Tomorrow: "what date is tomorrow?" / "whats tomorrows date?"
    if re.match(r"^what(?:s| is)?(?: the)? (?:day|date)(?: is)? tomorrow\s*[?.!]*$",
                t) or re.match(r"^what(?:s| is)? tomorrows date\s*[?.!]*$", t) \
            or re.match(r"^what day is tomorrow\s*[?.!]*$", t):
        return "Tomorrow is %s." % _fmt_date(today + datetime.timedelta(days=1))

    # Yesterday: "what date is yesterday?" / "what was yesterdays date?"
    if re.match(r"^what(?:s| is| was)?(?: the)? (?:day|date)(?: is| was)? yesterday\s*[?.!]*$",
                t) or re.match(r"^what(?:s| is| was)? yesterdays date\s*[?.!]*$", t) \
            or re.match(r"^what day was yesterday\s*[?.!]*$", t):
        return "Yesterday was %s." % _fmt_date(today - datetime.timedelta(days=1))

    # Today: "what day is it?" / "what is the date?" / "whats todays date?"
    # ("what date is it today?" carries both "is it" AND "today".)
    if re.match(r"^what(?:s| is)?(?: the)? (?:day|date)(?: is it)?(?: today)?\s*[?.!]*$",
                t) or re.match(r"^what(?:s| is)? todays date\s*[?.!]*$", t) \
            or re.match(r"^todays date\s*[?.!]*$", t) \
            or re.match(r"^what day is (?:it|today)\s*[?.!]*$", t):
        return "Today is %s." % _fmt_date(today)

    return None


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
        # SVC-P0-2: no search backend exists; this handler stays only
        # for the livetools gate harness, which calls exec_tool
        # directly. The model is never offered search (see TOOLS).
        return error_block(name, arg, "search_unavailable",
                           "no search backend is configured on this "
                           "deployment. Do not answer from memory; tell "
                           "the user search is unavailable."), \
            {"ok": False, "status": "unavailable",
             "detail": "no search backend is configured on this deployment"}
    if name == "fetch":
        if not arg.startswith(("https://", "http://")) or not _public_url(arg):
            return error_block(name, arg, "blocked_url",
                               "only public http(s) URLs can be fetched. "
                               "Do not answer from memory; tell the user "
                               "the URL cannot be fetched."), \
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
            return tool_block(name, arg,
                              truncate_to_budget(text, MAX_FETCH_TOKENS)), \
                {"ok": True, "status": f"{status} OK", "detail": None}
        except Exception as exc:
            # SVC-P2-9: the raw exception (local paths, resolver errors)
            # used to go into the model-facing block AND the UI detail.
            # The block and the UI get a generic message; the detail
            # stays in the server log.
            _log(f"[fetch] failed {arg}: {type(exc).__name__}: {exc}")
            return error_block(name, arg, "fetch_failed",
                               "the page could not be fetched. Do not "
                               "answer from memory; tell the user the page "
                               "could not be fetched."), \
                {"ok": False, "status": "fetch_failed",
                 "detail": "the page could not be fetched"}
    if name == "read_file":
        try:
            path = (DATA_DIR / arg).resolve()
            if DATA_DIR.resolve() not in path.parents and path != DATA_DIR.resolve():
                return error_block(name, arg, "blocked_path",
                                   "files are limited to the data directory. "
                                   "Do not answer from memory; tell the user "
                                   "the file cannot be read."), \
                    {"ok": False, "status": "blocked_path",
                     "detail": "files are limited to the data directory"}
            if not path.is_file():
                return error_block(name, arg, "not_found",
                                   "no such file in the data directory. Do "
                                   "not answer from memory; tell the user "
                                   "the file was not found."), \
                    {"ok": False, "status": "not_found",
                     "detail": "no such file in the data directory"}
            return tool_block(name, arg,
                              truncate_to_budget(path.read_text("utf-8", "replace"))), \
                {"ok": True, "status": "ok", "detail": None}
        except Exception as exc:
            _log(f"[read_file] failed {arg}: {type(exc).__name__}: {exc}")
            return error_block(name, arg, "read_failed",
                               "the file could not be read. Do not answer "
                               "from memory; tell the user the file could "
                               "not be read."), \
                {"ok": False, "status": "read_failed",
                 "detail": "the file could not be read"}
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
        try:
            expr = detect_direct_arithmetic(messages[-1].get("content", ""))
        except Exception:
            expr = None
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
                        "transcript": prompt + reply,
                        "truncated": False}
    # Deterministic date layer (extended 2026-11-11): a direct date
    # question as the last user message is answered without generation.
    # Disjoint from the arithmetic patterns above; any detection or
    # parsing failure falls through to the model (SVC-P1-4: the date
    # hook used to call detect_direct_date unwrapped, so one date-layer
    # bug turned a date question into a failed turn instead of a model
    # answer — it now has the same try/except as the calc hook).
    if messages and messages[-1].get("role") == "user":
        try:
            date_reply = detect_direct_date(messages[-1].get("content", ""),
                                            messages)
        except Exception:
            date_reply = None
        if date_reply is not None:
            prompt = render_prompt(messages)
            return {"reply": date_reply,
                    "tool_calls": [{"tool": "date",
                                    "arg": messages[-1].get("content", "").strip(),
                                    "deterministic": True}],
                    "transcript": prompt + date_reply,
                    "truncated": False}
    prompt = render_prompt(messages)
    # SVC-P2-14: generate() silently drops the oldest turns when the
    # prompt exceeds the context window. Flag it so the caller can say
    # so instead of answering from a silently shortened history.
    truncated = count_tokens(prompt) > (N_CTX - CTX_RESERVE)
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
    if find_complete_call(reply, eof=True):
        # SVC-P2-12: tool-loop exhaustion — every round ended in a call
        # line and no final answer was ever generated. Don't leak the
        # raw call syntax to the user as the reply.
        reply = ("I couldn't complete that request — the tool calls "
                 "didn't lead to an answer. Please try rephrasing.")
    return {"reply": visible_text(reply), "tool_calls": calls,
            "transcript": transcript, "truncated": truncated}


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


class _BodyTooLarge(Exception):
    """Raised by _body() when Content-Length exceeds the 1 MB cap."""


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
        if n < 0:
            raise ValueError("negative Content-Length")
        if n > 1_000_000:
            # SVC-P2-5: unbounded reads let a huge body OOM the handler
            # thread. 1 MB is far above any real chat payload.
            raise _BodyTooLarge()
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = urlsplit(self.path).path  # SVC-P2-7: match the parsed
        # path, not the raw path — a query string must not 404 a route.
        if path == "/health":
            if not STATE["ready"]:
                # SVC-P2-1: a loading/failed service is not healthy;
                # return 503 so status-based monitors (and Render's own
                # health checks) see the truth. The body is unchanged,
                # and ensure_node.sh/run_node.sh grep the body, so they
                # are unaffected.
                return self._json(503, {"status": "loading" if not STATE["error"] else "error",
                                        "model": MODEL_LABEL, "detail": STATE["error"]})
            return self._json(200, {
                "status": "ok", "model": MODEL_LABEL, "model_type": "nae1",
                "tokenizer_vocab": 32000, "context": N_CTX,
                "params": 100111872, "step": STEP, "heldout_ppl": GUARDRAIL_PPL,
                "engine": f"torch (native Nae1ForCausalLM, {_DTYPE_NAME})",
                "device": DEVICE})
        if path == "/diag":
            if not DIAG_KEY or not hmac.compare_digest(_bearer(self), DIAG_KEY):
                return self._json(401, {"error": "unauthorized"})
            return self._json(200, {"ready": STATE["ready"], "error": STATE["error"],
                                    "uptime_s": round(time.time() - STATE["started"]),
                                    "model": MODEL_LABEL, "hf_source": f"{HF_REPO}/{HF_SUBDIR}"})
        v1path = path
        if v1path == "/v1/models":
            # SVC-P2-2 (conscious decision): the model list is PUBLIC.
            # It reveals only the model id, and OpenAI-compatible
            # clients list models before/after auth setup — a 401 here
            # breaks them. The single-model lookup below is public for
            # the same reason; generation still requires the token.
            return self._json(200, {"object": "list",
                                    "data": [self._v1_model_object()]}, cors=True)
        if v1path.startswith("/v1/models/"):
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
        if v1path not in ("/chat", "/generate", "/abort") and not is_v1:
            return self._json(404, {"error": "not found"},
                              cors=v1path.startswith("/v1/"))
        if not _chat_auth_ok(self):
            return self._json(401, {"error": "unauthorized"}, cors=is_v1)
        if v1path == "/abort":
            # SVC-P2-18: a no-op BY DESIGN. Generations are short and
            # synchronous here, so there is never one in flight to stop;
            # cancellation is client-side (the browser aborts the fetch
            # and the dead socket raises ClientGone mid-generation).
            # The 200 keeps the site Worker's /abort forward from 502ing.
            return self._json(200, {"ok": True, "aborted": 0})
        if not STATE["ready"]:
            return self._json(503, {"error": "model not ready",
                                    "detail": STATE["error"]}, cors=is_v1)
        try:
            req = self._body()
        except _BodyTooLarge:
            if is_v1:
                return self._v1_error(413, "request body too large",
                                      "invalid_request_error")
            return self._json(413, {"error": "request body too large"})
        except Exception:
            return self._json(400, {"error": "invalid json"}, cors=is_v1)
        # SVC-P1-1: parameter parsing used to sit OUTSIDE the try above,
        # so a wrong-typed field ("temperature": "hot", "max_tokens":
        # "many", "top_p": null) raised uncaught and dropped the
        # connection with no HTTP response. Parse inside try/except and
        # answer 400; clamp every value to a sane range (max_new >= 1,
        # finite sampling params — NaN used to pass straight through to
        # the sampler).
        try:
            max_new = int(req.get("max_tokens",
                                  req.get("max_new_tokens", 160)))
            temperature = float(req.get("temperature", 0.7))
            top_p = float(req.get("top_p", 0.9))
            rep = float(req.get("repetition_penalty", 1.0))
            min_p = float(req.get("min_p", 0.0))
            if not all(math.isfinite(v) for v in
                       (temperature, top_p, rep, min_p)):
                raise ValueError("non-finite sampling parameter")
        except (TypeError, ValueError):
            if is_v1:
                return self._v1_error(400, "invalid sampling parameters",
                                      "invalid_request_error")
            return self._json(400, {"error": "invalid sampling parameters"})
        max_new = max(1, min(max_new, 512))
        temperature = min(max(temperature, 0.0), 2.0)
        top_p = min(max(top_p, 0.01), 1.0)
        rep = min(max(rep, 0.5), 2.0)
        min_p = min(max(min_p, 0.0), 0.5)
        if is_v1:
            return self._v1_chat_completions(req, max_new, temperature,
                                             top_p, rep, min_p)
        if v1path == "/generate":
            with GEN_LOCK:
                text = generate(str(req.get("prompt", "")), max_new,
                                temperature, top_p, stop_on_call=False,
                                rep_penalty=rep, min_p=min_p)
            return self._json(200, {"text": text})
        # SVC-P2-6: the old plain-JSON /chat branch below this point was
        # unreachable — every request with "prompt" or "messages" took
        # the stream branch, so /chat ALWAYS streamed SSE. The dead
        # branch is removed; /chat streams, and a request with neither
        # field is a 400 (the old branch's only reachable behavior).
        if "prompt" not in req and "messages" not in req:
            return self._json(400, {"error": "messages required"})
        return self._site_stream(req, max_new, temperature, top_p, rep, min_p)

    def _v1_chat_completions(self, req, max_new, temperature, top_p, rep, min_p):
        """OpenAI-compatible chat completion.

        Request: {model?, messages: [{role, content}], temperature?,
        top_p?, max_tokens?, repetition_penalty?, min_p?, stream?}.
        Unknown fields (e.g. response_format) are ignored. The served
        model is fixed, so a model name that is not ours is a 404, not
        a silent substitution. stream=true returns OpenAI-style SSE
        chunks driven incrementally from chat(..., emit=...) — the
        first chunk arrives after the first generation step, not after
        the whole reply (SVC-P1-2).
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
        if req.get("stream"):
            return self._v1_stream_live(convo, max_new, temperature,
                                        top_p, rep, min_p)
        try:
            out = chat(convo, max_new, temperature, top_p,
                       rep_penalty=rep, min_p=min_p)
        except Exception as exc:
            return self._v1_error(500, f"generation failed: {exc}",
                                  "internal_error")
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
            # NeuralAI extension (SVC-P2-14): true when the prompt was
            # truncated to fit the context window.
            "truncated": out.get("truncated", False),
        }, cors=True)

    def _v1_stream_live(self, convo, max_new, temperature, top_p, rep, min_p):
        """Incremental OpenAI-compatible SSE stream (SVC-P1-2).

        Headers go out first; chat(..., emit=...) drives the content
        chunks token by token, so a slow CPU share no longer means
        tens of seconds of silence before the first byte (proxies and
        clients used to time out on that silence). Every frame ends
        with "\\n\\n" — the old single-"\\n" framing never dispatched
        in spec-compliant SSE decoders. tool_calls ride the final
        chunk's delta (SVC-P2-3, OpenAI chunk shape); a usage chunk
        closes the stream (SVC-P2-4, same shape as
        stream_options.include_usage). The emit callback receives
        cumulative visible text; only the delta since the last chunk
        is sent. Deterministic hooks (calc/date) return without ever
        calling emit, so their reply is sent whole after chat() returns.
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
        state = {"sent": "", "dead": False}

        def frame(obj):
            if state["dead"]:
                return
            try:
                self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                self.wfile.flush()
            except OSError:
                state["dead"] = True
                raise ClientGone()

        def chunk(delta, finish_reason):
            return {"id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": API_MODEL_ID,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish_reason}]}

        def send_text(text):
            # Emit `text` in <=48-char word-boundary chunks (not one
            # byte at a time, not one giant frame).
            buf = ""
            for word in str(text).split(" "):
                piece = word if not buf else " " + word
                if buf and len(buf) + len(piece) > 48:
                    frame(chunk({"content": buf}, None))
                    buf = word
                else:
                    buf += piece
            if buf:
                frame(chunk({"content": buf}, None))

        def emit(cumulative):
            if state["dead"]:
                raise ClientGone()
            # chat()'s emit is cumulative and monotone (visible_text of
            # a growing generation), so the unsent tail is a slice.
            delta = cumulative[len(state["sent"]):]
            if delta:
                state["sent"] = cumulative
                send_text(delta)
            if state["dead"]:
                raise ClientGone()

        try:
            frame(chunk({"role": "assistant", "content": ""}, None))
            out = chat(convo, max_new, temperature, top_p,
                       rep_penalty=rep, emit=emit, min_p=min_p)
            reply = out["reply"]
            if reply and not state["sent"].endswith(reply):
                # No emit covered this reply (deterministic hook, or a
                # tool-loop-exhaustion message) — send it whole.
                send_text(reply)
                state["sent"] += reply
            final = chunk({"tool_calls": out["tool_calls"]}
                          if out["tool_calls"] else {}, "stop")
            frame(final)
            prompt_tokens = count_tokens(render_prompt(convo))
            completion_tokens = count_tokens(state["sent"])
            frame({"id": cid, "object": "chat.completion.chunk",
                   "created": created, "model": API_MODEL_ID,
                   "choices": [],
                   "usage": {"prompt_tokens": prompt_tokens,
                             "completion_tokens": completion_tokens,
                             "total_tokens": prompt_tokens + completion_tokens},
                   "truncated": out.get("truncated", False)})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (OSError, ClientGone):
            return
        except Exception as exc:
            # Generation failed mid-stream: headers are already sent, so
            # a 500 is impossible — end the stream with [DONE] and log.
            _log(f"[v1-stream] error: {exc}")
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass

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
                # SVC-P1-2: "\n\n" after every frame — the old single
                # "\n" is not SSE-compliant (an event dispatches only
                # on a blank line). The site parser splits on "\n" and
                # skips empty lines, so this is wire-compatible with it.
                self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
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
            # "truncated" (SVC-P2-14) flags a silently shortened history.
            frame({"text": out["reply"], "done": True, "eos": True,
                   "tool_calls": out["tool_calls"],
                   "truncated": out.get("truncated", False)})
        except ClientGone:
            return
        except Exception as exc:
            _log(f"[site-stream] error: {exc}")
            frame({"text": state["last"], "done": True, "eos": False})


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # SVC-P1-5: log the effective TZ at startup — the date layer's
    # "today" fallback depends on it, and a silent host TZ used to be
    # able to shift every date answer by a day.
    _log(f"[tz] TZ={os.environ.get('TZ')} local-today="
         f"{datetime.date.today()} ({time.tzname})")
    if os.environ.get("BLOCKING_LOAD"):
        # Modal: bind the port only once the model is ready, so the
        # platform's startup wait covers the full cold start and no
        # request can land on a not-ready server.
        load_model()
    else:
        threading.Thread(target=load_model, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    _log(f"[listen] :{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
