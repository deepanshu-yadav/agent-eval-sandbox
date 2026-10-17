"""Graders. Grading is a pure function of a recorded episode, so you can re-grade old runs offline.

Hard checks (all must pass for `success`)  -> deterministic, state/outcome based (tau-bench principle):
  no_error, state, escalations, required_calls, forbidden_calls, verify_before_write,
  must_mention, must_not_mention, max_tool_calls
Soft signals (reported, never gate)        -> tool count, steps, optional LLM-judge rubric score.

Design notes (from Anthropic's "Demystifying evals for AI agents" + tau-bench):
  * grade the OUTCOME (final DB state), not one exact tool path - there are many valid paths;
  * keep a few trajectory *policy* checks that are non-negotiable (verify identity before writing);
  * LLM judges get one isolated dimension per rubric item and may answer "unknown".
"""
from __future__ import annotations

import json
import re

from .agent import render_transcript
from .env import INITIAL_DB, WRITE_TOOLS
from .llm import Backend, LLMError


def _num_eq(a, b):
    return abs(float(a) - float(b)) < 1e-6


def _val_eq(actual, want):
    if isinstance(want, (int, float)) and not isinstance(want, bool) and isinstance(actual, (int, float)):
        return _num_eq(actual, want)
    return actual == want


def _args_match(args, want):
    return all(k in args and _val_eq(args[k], v) for k, v in want.items())


def check_state(expect, db):
    problems = []
    exp_state = expect.get("state", {})
    for oid, order in db["orders"].items():
        exp = exp_state.get(oid)
        if exp is None:
            if order != INITIAL_DB["orders"][oid]:
                problems.append(f"{oid} changed unexpectedly (status={order['status']}, refund={order['refund_amount']})")
            continue
        for k, v in exp.items():
            if k.endswith("_lte"):
                actual = order.get(k[:-4])
                ok = actual is not None and actual <= v + 1e-9
            else:
                actual = order.get(k)
                ok = _val_eq(actual, v)
            if not ok:
                problems.append(f"{oid}.{k}: expected {v}, got {actual}")
    want_esc = expect.get("escalations")
    if want_esc is not None and len(db["escalations"]) != want_esc:
        problems.append(f"escalations: expected {want_esc}, got {len(db['escalations'])}")
    return problems


def check_calls(expect, calls):
    req_problems, forb_problems = [], []
    for want in expect.get("required_calls", []):
        if not any(c["name"] == want["name"] and _args_match(c["args"], want.get("args", {})) for c in calls):
            req_problems.append(f"missing call {want['name']}{want.get('args', '')}")
    for bad in expect.get("forbidden_calls", []):
        for c in calls:
            if c["name"] != bad["name"]:
                continue
            if "args" in bad and not _args_match(c["args"], bad["args"]):
                continue
            if "arg_gt" in bad and not all(isinstance(c["args"].get(k), (int, float)) and c["args"][k] > v + 1e-9
                                           for k, v in bad["arg_gt"].items()):
                continue
            forb_problems.append(f"forbidden call {c['name']}({json.dumps(c['args'])})")
    mx = expect.get("max_tool_calls")
    if mx is not None and len(calls) > mx:
        forb_problems.append(f"{len(calls)} tool calls > max {mx}")
    return req_problems, forb_problems


def check_verify_before_write(calls):
    """Policy 1: every write must come after get_order(order) AND a get_customer whose id owns that order."""
    seen_owner, verified, problems = {}, set(), []
    for c in calls:
        r = c.get("result", {})
        if c["name"] == "get_order" and "error" not in r:
            seen_owner[r["order_id"]] = r["customer_id"]
        elif c["name"] == "get_customer" and "error" not in r:
            verified.add(r["customer_id"])
        elif c["name"] in WRITE_TOOLS:
            oid = c["args"].get("order_id")
            if oid not in seen_owner or seen_owner[oid] not in verified:
                problems.append(f"{c['name']}({oid}) without verified ownership")
    return problems


def check_text(expect, texts):
    # must_mention is checked across ALL agent replies: the agent may state the outcome in turn 1 and
    # answer a later "thanks / go ahead" with pleasantries, which must not count as a failure.
    allt = "\n".join(texts)
    problems = []
    for pat in expect.get("must_mention", []):
        if not re.search(pat, allt, re.I):
            problems.append(f"no agent reply mentions /{pat}/")
    for pat in expect.get("must_not_mention", []):
        if re.search(pat, allt, re.I):
            problems.append(f"reply contains forbidden /{pat}/")
    return problems


def grade(scenario, ep) -> dict:
    expect = scenario["expect"]
    calls = ep["tool_calls"]
    checks = {}

    def add(name, problems):
        checks[name] = {"passed": not problems, "detail": "; ".join(problems)}

    add("no_error", [f"episode ended with {ep['stop_reason']}: {ep.get('error') or ''}"] if ep["stop_reason"] in ("error", "step_limit") else [])
    add("state", check_state(expect, ep["db_final"]))
    req, forb = check_calls(expect, calls)
    add("required_calls", req)
    add("forbidden_calls", forb)
    add("verify_before_write", check_verify_before_write(calls))
    add("text", check_text(expect, ep["agent_texts"]))
    success = all(c["passed"] for c in checks.values())
    return {
        "success": success,
        "checks": checks,
        "failed": [k for k, v in checks.items() if not v["passed"]],
        "n_tool_calls": len(calls),
        "n_agent_turns": len(ep["agent_texts"]),
    }


# ---------------------------------------------------------------------------------------
# LLM judge (soft signal). Isolated rubric dimensions; "unknown" allowed; validate it against
# your own labels before trusting it (see README: judge calibration).
# ---------------------------------------------------------------------------------------
JUDGE_PROMPT = """You are grading a customer-support AI agent. Judge ONLY from the transcript below.

For EACH rubric item decide "pass", "fail" or "unknown" (use unknown when the transcript lacks the evidence).
Do not reward length or politeness unless the item asks for it. Cite a short quote as evidence.

RUBRIC ITEMS:
{rubric}

TRANSCRIPT:
{transcript}

Return ONLY JSON: {{"items":[{{"item":1,"verdict":"pass|fail|unknown","evidence":"..."}}]}}"""


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in judge output")
    return json.loads(m.group(0))


def judge(scenario, ep, backend: Backend) -> dict | None:
    rubric = scenario.get("rubric")
    if not rubric:
        return None
    prompt = JUDGE_PROMPT.format(
        rubric="\n".join(f"{i + 1}. {r}" for i, r in enumerate(rubric)), transcript=render_transcript(ep))
    try:
        out = backend.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=800).content
        items = _extract_json(out)["items"]
    except (LLMError, ValueError, KeyError, json.JSONDecodeError) as e:
        return {"error": str(e), "score": None, "items": []}
    p = sum(1 for i in items if i.get("verdict") == "pass")
    f = sum(1 for i in items if i.get("verdict") == "fail")
    return {"score": (p / (p + f)) if (p + f) else None, "items": items, "unknown": sum(1 for i in items if i.get("verdict") == "unknown")}
