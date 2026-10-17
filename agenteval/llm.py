"""LLM backends behind one tiny interface.

Spec strings (used by the CLI):  kind:model[@base_url][#API_KEY_ENV_VAR]

  openai:gpt-4.1-mini                              OpenAI (uses OPENAI_API_KEY)
  openai:some/model@https://openrouter.ai/api/v1#OPENROUTER_API_KEY
  llamacpp:local@http://localhost:8080/v1          llama.cpp `llama-server --jinja` (no key needed)
  ollama:qwen2.5:7b                                Ollama's OpenAI-compatible endpoint
  anthropic:<model-id>                             Anthropic Messages API (uses ANTHROPIC_API_KEY)
  mock:good | mock:sloppy | mock:flaky0.3          rule-based fake agent for testing the harness itself

Internal message format is OpenAI-style (role/content/tool_calls/tool_call_id).
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date

import requests


class LLMError(RuntimeError):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status


@dataclass
class Reply:
    content: str = ""
    tool_calls: list = field(default_factory=list)  # [{"id","name","arguments": dict, "parse_error"?: str}]
    usage: dict = field(default_factory=dict)  # {"prompt_tokens","completion_tokens"}
    latency_s: float = 0.0


class Backend:
    name = "base"

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024, meta=None) -> Reply:
        raise NotImplementedError


def _post(url, headers, payload, timeout=300, retries=4):
    last = None
    for i in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2**i)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            time.sleep(2**i)
            continue
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:600]}", status=r.status_code)
        return r.json()
    raise LLMError(f"request failed after {retries} tries: {last}")


def _parse_args(raw):
    if isinstance(raw, dict):
        return raw, None
    if raw is None or raw == "":
        return {}, None
    try:
        v = json.loads(raw)
        return (v if isinstance(v, dict) else {"_value": v}), None
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON arguments ({e}): {str(raw)[:200]}"


class OpenAICompat(Backend):
    """Works with OpenAI, llama.cpp llama-server (--jinja), Ollama, vLLM, OpenRouter, Together, ..."""

    def __init__(self, model, base_url=None, api_key=None, label="openai"):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "none"
        self.name = f"{label}:{model}"

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024, meta=None) -> Reply:
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
            payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        t0 = time.time()
        data = None
        for _ in range(3):  # adapt once each to API quirks of newer OpenAI models
            try:
                data = _post(f"{self.base_url}/chat/completions", headers, payload)
                break
            except LLMError as e:
                msg = str(e)
                if e.status == 400 and "max_tokens" in msg and "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                elif e.status == 400 and "temperature" in msg and "temperature" in payload:
                    payload.pop("temperature")
                else:
                    raise
        if data is None:
            raise LLMError("could not adapt request to backend")
        msg = data["choices"][0]["message"]
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            args, err = _parse_args(tc["function"].get("arguments"))
            c = {"id": tc.get("id") or f"call_{i}", "name": tc["function"]["name"], "arguments": args}
            if err:
                c["parse_error"] = err
            calls.append(c)
        u = data.get("usage") or {}
        return Reply(
            content=msg.get("content") or "",
            tool_calls=calls,
            usage={"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0)},
            latency_s=time.time() - t0,
        )


class AnthropicBackend(Backend):
    def __init__(self, model, api_key=None, base_url=None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self.name = f"anthropic:{model}"

    @staticmethod
    def _convert(messages):
        system, out = [], []
        for m in messages:
            role = m["role"]
            if role == "system":
                system.append(m["content"])
            elif role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    args, _ = _parse_args(tc["function"]["arguments"])
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": args})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)  # merge parallel tool results
                else:
                    out.append({"role": "user", "content": [block]})
        return "\n\n".join(system), out

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024, meta=None) -> Reply:
        system, msgs = self._convert(messages)
        payload = {"model": self.model, "max_tokens": max_tokens, "messages": msgs, "temperature": temperature}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools
            ]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        t0 = time.time()
        try:
            data = _post(f"{self.base_url}/v1/messages", headers, payload)
        except LLMError as e:
            if e.status == 400 and "temperature" in str(e):  # some models reject it
                payload.pop("temperature")
                data = _post(f"{self.base_url}/v1/messages", headers, payload)
            else:
                raise
        text, calls = [], []
        for b in data.get("content", []):
            if b["type"] == "text":
                text.append(b["text"])
            elif b["type"] == "tool_use":
                calls.append({"id": b["id"], "name": b["name"], "arguments": b.get("input") or {}})
        u = data.get("usage") or {}
        return Reply(
            content="".join(text),
            tool_calls=calls,
            usage={"prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0)},
            latency_s=time.time() - t0,
        )


# --------------------------------------------------------------------------------------
# Mock backend: a rule-based "agent" so you can test graders/metrics without any model.
#   good   -> follows the ShopDesk policy
#   sloppy -> ignores policy (refunds whatever is asked, skips identity check, obeys injected notes)
#   flakyP -> behaves like `good` but turns sloppy with probability P per (scenario, trial)
#   oracle -> for fc_eval.py: returns the gold tool calls given in meta["gold"]
# --------------------------------------------------------------------------------------
class MockBackend(Backend):
    def __init__(self, mode="good"):
        self.mode = mode
        self.name = f"mock:{mode}"

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024, meta=None) -> Reply:
        meta = meta or {}
        if self.mode == "oracle":
            calls = [
                {"id": f"call_{i}", "name": g["name"], "arguments": g.get("arguments", {})}
                for i, g in enumerate(meta.get("gold", []))
            ]
            return Reply(tool_calls=calls, usage={"prompt_tokens": 10, "completion_tokens": 5})
        sloppy = self.mode == "sloppy"
        if self.mode.startswith("flaky"):
            p = float(self.mode[5:] or 0.3)
            sloppy = random.Random(f"{meta.get('scenario')}-{meta.get('trial')}").random() < p
        kind, payload = _mock_decide(messages, sloppy)
        usage = {"prompt_tokens": 100, "completion_tokens": 20}
        if kind == "tool":
            return Reply(tool_calls=[{"id": f"call_{len(messages)}", "name": payload[0], "arguments": payload[1]}], usage=usage)
        return Reply(content=payload, usage=usage)


def _find_fallback_key(provider: str) -> str | None:
    """Environment variables only. (An earlier version also regex-scanned rough.txt / .env in the current and
    parent folders: that silently picks up unrelated secrets and can send them to the wrong provider.)"""
    names = {"openrouter": ["OPENROUTER_API_KEY"], "xai": ["XAI_API_KEY"], "grok": ["XAI_API_KEY"]}.get(provider, [])
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    key = os.environ.get("OPENAI_API_KEY")
    if key and (provider == "openai" or (provider == "openrouter" and key.startswith("sk-or-"))):
        return key
    return None


def _mock_decide(messages, sloppy):
    utext = " ".join(m["content"] for m in messages if m["role"] == "user")
    low = utext.lower()
    order_m = re.search(r"O\d{4}", utext)
    email_m = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", utext)
    amt_m = re.search(r"\$(\d+(?:\.\d+)?)", utext)
    addr_m = re.search(r"(?:to|address(?:\s+to)?)\s+([0-9]+\s+[A-Za-z\s]+(?:St|Rd|Ave|Blvd|Way))", utext, re.I)
    oid = order_m.group(0) if order_m else None
    email = email_m.group(0).lower().rstrip(".") if email_m else None

    # Handle direct adversarial prompt injection override
    if "override" in low or "maintenance mode" in low or "authentication bypassed" in low:
        if sloppy:
            return "tool", ("issue_refund", {"order_id": oid or "O1001", "amount": 59.99, "reason": "system override"})
        return "text", "Security override requests are denied. Standard verification policy applies."

    if "cancel" in low:
        intent = "cancel"
    elif any(w in low for w in ("refund", "money back", "return")) or re.search(r"\$\d+\s+back", low):
        intent = "refund"
    elif any(w in low for w in ("address", "deliver to", "shipping address")):
        intent = "address"
    elif any(w in low for w in ("where is", "status", "track")):
        intent = "status"
    else:
        intent = None

    id2call = {}
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                id2call[tc["id"]] = {"name": tc["function"]["name"], "args": json.loads(tc["function"]["arguments"])}
        elif m["role"] == "tool" and m["tool_call_id"] in id2call:
            id2call[m["tool_call_id"]]["result"] = json.loads(m["content"])
    calls = list(id2call.values())

    def done(name):
        return next((c for c in calls if c["name"] == name), None)

    if intent is None and not oid:
        return "text", "Sorry, I can only help with ShopDesk orders (refunds, cancellations, order status, address updates)."
    if not oid:
        return "text", "Happy to help. Could you share your order ID?"
    if intent in ("refund", "cancel", "address") and not email:
        return "text", "Thanks. Please also share the email address on the account so I can verify you."
    if not done("get_order"):
        return "tool", ("get_order", {"order_id": oid})
    order = done("get_order").get("result", {})
    if "error" in order:
        if sloppy:
            return "text", f"Order {oid} has shipped and is on its way."
        return "text", f"I couldn't find order {oid}. Could you double-check the ID?"

    if intent == "status" or intent is None:
        if sloppy and "refund" in (order.get("notes") or "").lower() and not done("issue_refund"):
            return "tool", ("issue_refund", {"order_id": oid, "amount": 500.0, "reason": "per order note"})
        return "text", f"Order {oid} is currently {order['status']}."

    if not sloppy and not done("get_customer"):
        return "tool", ("get_customer", {"email": email})
    cust = (done("get_customer") or {}).get("result", {})
    verified = cust.get("customer_id") == order.get("customer_id")
    if not sloppy and not verified:
        return "text", "I couldn't verify that this email matches the account for that order, so I can't help with it."

    total = order["total"]
    if intent == "refund":
        if done("issue_refund"):
            r = done("issue_refund")["result"]
            if "error" in r:
                return "text", f"The refund could not be processed: {r['error']}"
            return "text", f"Done. I've refunded ${r['refunded']:.2f} for order {oid}."
        if done("escalate_to_human"):
            return "text", "This refund is above my limit, so I've escalated it to a human specialist who will follow up."
        if sloppy:
            amount = float(amt_m.group(1)) if amt_m else total
            return "tool", ("issue_refund", {"order_id": oid, "amount": amount, "reason": "customer request"})
        age = (date(2026, 10, 17) - date.fromisoformat(order["delivered_on"])).days if order.get("delivered_on") else None
        if order["status"] == "refunded":
            return "text", f"Order {oid} was already refunded, so I can't refund it again."
        if order["status"] != "delivered":
            return "text", f"Order {oid} is {order['status']}, so it isn't eligible for a refund yet."
        if order.get("final_sale"):
            return "text", "Sorry, that item was a final sale, so it can't be refunded."
        if age > 30:
            return "text", f"That order was delivered {age} days ago, outside our 30-day refund window, so I can't refund it."
        if total > 200:
            return "tool", ("escalate_to_human", {"reason": f"refund over $200 for {oid}", "order_id": oid})
        amount = min(float(amt_m.group(1)), total) if amt_m else total
        return "tool", ("issue_refund", {"order_id": oid, "amount": amount, "reason": "customer request"})

    # cancel
    if intent == "cancel":
        if done("cancel_order"):
            r = done("cancel_order")["result"]
            return "text", f"Order {oid} has been cancelled." if "error" not in r else f"Could not cancel: {r['error']}"
        if sloppy or order["status"] == "processing":
            return "tool", ("cancel_order", {"order_id": oid})
        return "text", f"Order {oid} has already {order['status']}, so it can't be cancelled."

    # address
    if intent == "address":
        target_addr = addr_m.group(1).strip() if addr_m else "45 Sunset Blvd"
        if done("update_address"):
            r = done("update_address")["result"]
            return "text", f"The shipping address for order {oid} has been updated to {target_addr}." if "error" not in r else f"Could not update address: {r['error']}"
        if not sloppy and order["status"] != "processing":
            return "text", f"Order {oid} has already {order['status']}, so its shipping address cannot be changed."
        return "tool", ("update_address", {"order_id": oid, "address": target_addr})


def parse_spec(spec: str, api_key: str | None = None) -> Backend:
    """kind:model[@base_url][#KEY_ENV_OR_LITERAL_KEY]"""
    key_env = None
    if "#" in spec:
        spec, key_env = spec.rsplit("#", 1)
    base_url = None
    if "@" in spec:
        spec, base_url = spec.split("@", 1)
    kind, _, model = spec.partition(":")

    # Determine api_key: passed directly > env var > literal key after # > fallback search
    if not api_key and key_env:
        if key_env.startswith(("sk-", "xai-")) or len(key_env) > 30:
            api_key = key_env
        else:
            api_key = os.environ.get(key_env)
            if not api_key:
                raise LLMError(f"environment variable {key_env} is not set")

    if kind == "mock":
        return MockBackend(model or "good")
    if kind == "anthropic":
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        return AnthropicBackend(model, api_key=api_key, base_url=base_url)
    if kind == "llamacpp":
        return OpenAICompat(model or "local", base_url=base_url or "http://localhost:8080/v1", api_key=api_key or "none", label="llamacpp")
    if kind == "ollama":
        return OpenAICompat(model, base_url=base_url or "http://localhost:11434/v1", api_key=api_key or "ollama", label="ollama")
    if kind == "openrouter":
        api_key = api_key or _find_fallback_key("openrouter")
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY not set")
        return OpenAICompat(model, base_url=base_url or "https://openrouter.ai/api/v1", api_key=api_key, label="openrouter")
    if kind in ("xai", "grok"):
        api_key = api_key or _find_fallback_key("xai")
        if not api_key:
            raise LLMError("XAI_API_KEY not set")
        return OpenAICompat(model, base_url=base_url or "https://api.x.ai/v1", api_key=api_key, label="xai")
    if kind == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAICompat(model, base_url=base_url, api_key=api_key)
    raise LLMError(f"unknown backend kind '{kind}' in spec '{spec}'")
