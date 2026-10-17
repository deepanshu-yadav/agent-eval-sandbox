#!/usr/bin/env python3
"""Single-turn function-calling eval on a *public* dataset (xLAM-style schema) - the cheap, fast layer
under the multi-turn ShopDesk eval. Measures: valid call format, function-name accuracy, argument accuracy,
exact match of the whole call set.

Data sources
  --file data/fc_sample.jsonl                 tiny bundled sample (offline smoke test)
  --hf Salesforce/xlam-function-calling-60k   needs `pip install datasets`, `huggingface-cli login`, and accepting the
                                              dataset terms on the Hub (gated; license CC-BY-NC-4.0 - research use)
  --hf Team-ACE/ToolACE                       other schema: convert it to the xLAM jsonl layout first (see README)

Each row: {"query": str, "tools": "<json list>", "answers": "<json list of {name, arguments}>"}
For BFCL itself (irrelevance, multi-turn, agentic web-search/memory) use the official `bfcl-eval` package instead.

Examples
  python fc_eval.py --agent llamacpp:local@http://localhost:8080/v1 --file data/fc_sample.jsonl
  python fc_eval.py --agent openai:gpt-4.1-mini --hf Salesforce/xlam-function-calling-60k --limit 200
  python fc_eval.py --agent mock:oracle --file data/fc_sample.jsonl
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from agenteval.llm import LLMError, parse_spec

_TYPES = {"str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number",
          "number": "number", "bool": "boolean", "boolean": "boolean", "dict": "object", "object": "object"}


def _json_type(t):
    t = str(t).lower().strip()
    if t.startswith(("list", "array", "tuple")):
        return {"type": "array"}
    for k, v in _TYPES.items():
        if t.startswith(k):
            return {"type": v}
    return {}  # unknown -> unconstrained


def to_openai_tool(t):
    """xLAM tool (flat {param: {type, description, default?}}) -> OpenAI-style function schema."""
    props, required = {}, []
    params = t.get("parameters", {})
    if "properties" in params:  # already JSON-schema
        return {"name": t["name"], "description": t.get("description", ""), "parameters": params}
    for k, spec in params.items():
        props[k] = {**_json_type(spec.get("type", "")), "description": spec.get("description", "")}
        if "default" not in spec and "optional" not in str(spec.get("type", "")).lower():
            required.append(k)
    return {"name": t["name"], "description": t.get("description", ""),
            "parameters": {"type": "object", "properties": props, "required": required}}


def _norm(v):
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return float(s)
        except ValueError:
            return s
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def score_row(pred, gold):
    """pred/gold: list of {"name","arguments"}. Order-insensitive matching of calls."""
    remaining = list(gold)
    name_hits = arg_hits = 0
    for p in pred:
        for g in remaining:
            if p["name"] == g["name"]:
                name_hits += 1
                if _norm(p["arguments"]) == _norm(g["arguments"]):
                    arg_hits += 1
                remaining.remove(g)
                break
    n = max(len(gold), 1)
    return {"name_acc": name_hits / n, "arg_acc": arg_hits / n,
            "exact": float(len(pred) == len(gold) and arg_hits == len(gold))}


def load_rows(a):
    if a.file:
        rows = [json.loads(line) for line in open(a.file) if line.strip()]
    else:
        from datasets import load_dataset  # pip install datasets
        ds = load_dataset(a.hf, split=a.split)
        rows = [dict(r) for r in ds.select(range(min(a.limit or len(ds), len(ds))))]
    return rows[: a.limit] if a.limit else rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--hf")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/fc_eval.jsonl")
    a = ap.parse_args()

    backend = parse_spec(a.agent)
    rows = load_rows(a)

    def work(row):
        tools = [to_openai_tool(t) for t in json.loads(row["tools"])]
        gold = json.loads(row["answers"])
        try:
            rep = backend.chat(
                [{"role": "system", "content": "Call the appropriate function(s) for the user's request. Do not answer in prose."},
                 {"role": "user", "content": row["query"]}],
                tools=tools, meta={"gold": gold})
            pred = [{"name": c["name"], "arguments": c["arguments"]} for c in rep.tool_calls if "parse_error" not in c]
            bad = sum(1 for c in rep.tool_calls if "parse_error" in c)
            s = score_row(pred, gold)
            s.update(valid_format=float(bool(rep.tool_calls) and not bad), error=None)
        except LLMError as e:
            pred, s = [], {"name_acc": 0, "arg_acc": 0, "exact": 0, "valid_format": 0, "error": str(e)}
        return {"query": row["query"], "gold": gold, "pred": pred, **s}

    with ThreadPoolExecutor(a.workers) as pool:
        out = list(pool.map(work, rows))
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n = len(out)
    avg = lambda k: sum(r[k] for r in out) / n
    print(f"agent={backend.name} rows={n}")
    print(f"  valid_format  {avg('valid_format'):.3f}\n  function-name {avg('name_acc'):.3f}\n  argument-acc  {avg('arg_acc'):.3f}\n  exact-match   {avg('exact'):.3f}")
    errs = sum(1 for r in out if r["error"])
    if errs:
        print(f"  ({errs} rows hit backend errors, counted as failures)")
    print(f"per-row details: {a.out}")


if __name__ == "__main__":
    main()
