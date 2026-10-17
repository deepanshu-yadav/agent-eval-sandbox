"""Run scenarios x trials in parallel, grade, aggregate, write JSONL + summary + markdown report."""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agent import make_user, run_episode
from .graders import grade, judge
from .metrics import aggregate


def load_scenarios(path, ids=None, categories=None):
    with open(path) as f:
        sc = json.load(f)
    if ids:
        sc = [s for s in sc if any(i in s["id"] for i in ids)]
    if categories:
        sc = [s for s in sc if s["category"] in categories]
    return sc


import threading

def run_suite(scenarios, agent, trials=3, user_backend=None, judge_backend=None, workers=4,
              max_turns=8, temperature=0.0, outdir="results", tag=None, progress=True, live=True):
    run_id = tag or time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(outdir, run_id)
    os.makedirs(outdir, exist_ok=True)
    jobs = [(s, t) for s in scenarios for t in range(trials)]
    results = []
    trials_path = os.path.join(outdir, "trials.jsonl")
    trials_file = open(trials_path, "w", encoding="utf-8")

    print_lock = threading.Lock()
    file_lock = threading.Lock()
    done_counter = 0

    def make_event_logger(s, t):
        s_id = s["id"]
        def on_event(etype, data):
            if not live or not progress:
                return
            with print_lock:
                if etype == "start":
                    print(f"[{s_id} #{t}] >>> START episode", flush=True)
                elif etype == "user":
                    txt = data["content"].replace("\n", " ")
                    if len(txt) > 130:
                        txt = txt[:127] + "..."
                    print(f"[{s_id} #{t}] [USER] {txt}", flush=True)
                elif etype == "tool_call":
                    args_s = json.dumps(data.get("arguments", {}))
                    if len(args_s) > 100:
                        args_s = args_s[:97] + "..."
                    print(f"[{s_id} #{t}] [TOOL CALL] {data['name']}({args_s})", flush=True)
                elif etype == "tool_result":
                    res_s = json.dumps(data.get("result", {}))
                    if len(res_s) > 100:
                        res_s = res_s[:97] + "..."
                    print(f"[{s_id} #{t}] [TOOL RES]  {res_s}", flush=True)
                elif etype == "agent":
                    txt = data["content"].replace("\n", " ")
                    if len(txt) > 130:
                        txt = txt[:127] + "..."
                    print(f"[{s_id} #{t}] [AGENT] {txt}", flush=True)
        return on_event

    def work(s, t):
        ev_cb = make_event_logger(s, t) if live else None
        ep = run_episode(s, agent, make_user(s, user_backend), trial=t, max_turns=max_turns, temperature=temperature, on_event=ev_cb)
        g = grade(s, ep)
        j = judge(s, ep, judge_backend) if judge_backend else None
        res = {"scenario_id": s["id"], "category": s["category"], "trial": t, "grade": g, "judge": j, "episode": ep}
        with file_lock:
            trials_file.write(json.dumps(res, ensure_ascii=False) + "\n")
            trials_file.flush()
        return res

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(work, s, t): (s["id"], t) for s, t in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                done_counter += 1
                if progress:
                    mark = "PASS" if r["grade"]["success"] else "FAIL " + ",".join(r["grade"]["failed"])
                    with print_lock:
                        print(f"[{done_counter}/{len(jobs)}] {r['scenario_id']} trial {r['trial']}: {mark} ({r['episode']['latency_s']:.2f}s)", flush=True)
    except KeyboardInterrupt:
        with print_lock:
            print(f"\n[INTERRUPT] Evaluation interrupted by user. Summarizing {len(results)} finished trial(s)...", flush=True)
    finally:
        trials_file.close()

    results.sort(key=lambda r: (r["scenario_id"], r["trial"]))
    if results:
        summary = summarize(results, scenarios, trials, agent.name)
        summary["run_id"] = run_id
        with open(os.path.join(outdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(outdir, "report.md"), "w") as f:
            f.write(render_report(summary))
        return summary, outdir
    return {}, outdir


def summarize(results, scenarios, trials, agent_name):
    per = defaultdict(lambda: [0, 0])  # id -> [n, c]
    cat_of = {s["id"]: s["category"] for s in scenarios}
    fail_reasons = defaultdict(Counter)
    for r in results:
        per[r["scenario_id"]][0] += 1
        per[r["scenario_id"]][1] += int(r["grade"]["success"])
        for k in r["grade"]["failed"]:
            fail_reasons[r["scenario_id"]][k] += 1
    ks = list(range(1, trials + 1))
    per_t = {k: tuple(v) for k, v in per.items()}
    overall = aggregate(per_t, ks)
    by_cat = {}
    for cat in sorted(set(cat_of.values())):
        by_cat[cat] = aggregate({k: v for k, v in per_t.items() if cat_of[k] == cat}, [1, trials] if trials > 1 else [1])
    n = len(results)
    jscores = [r["judge"]["score"] for r in results if r.get("judge") and r["judge"].get("score") is not None]
    ep = [r["episode"] for r in results]
    return {
        "agent": agent_name, "trials_per_task": trials, "n_tasks": len(per), "n_episodes": n,
        "overall": overall,
        "by_category": by_cat,
        "per_task": {k: {"n": v[0], "successes": v[1], "category": cat_of[k],
                         "failed_checks": dict(fail_reasons[k])} for k, v in per.items()},
        "avg_tool_calls": sum(r["grade"]["n_tool_calls"] for r in results) / n,
        "avg_agent_turns": sum(r["grade"]["n_agent_turns"] for r in results) / n,
        "avg_latency_s": sum(e["latency_s"] for e in ep) / n,
        "tokens": {"prompt": sum(e["usage"]["prompt_tokens"] for e in ep), "completion": sum(e["usage"]["completion_tokens"] for e in ep)},
        "judge_mean_score": (sum(jscores) / len(jscores)) if jscores else None,
        "infra_errors": sum(1 for e in ep if e["stop_reason"] == "error"),
    }


def render_report(s):
    L = [f"# Agent eval report: `{s['agent']}`", "",
         f"{s['n_tasks']} tasks x {s['trials_per_task']} trials = {s['n_episodes']} episodes. Infra errors: {s['infra_errors']}.", "",
         "## Headline (success = every hard check passes)", "", "| k | pass@k (any of k succeeds) | pass^k (all k succeed) |", "|---|---|---|"]
    for k, v in s["overall"].items():
        L.append(f"| {k} | {v['pass@k']:.2f} | {v['pass^k']:.2f} |")
    L += ["", "A big gap between pass@1 and pass^k means the agent *can* do the task but is not *reliable*.", "",
          "## By category", "", "| category | pass^1 | pass^k (k=n) |", "|---|---|---|"]
    for cat, d in s["by_category"].items():
        last = d[max(d)]
        L.append(f"| {cat} | {d[1]['pass^k']:.2f} | {last['pass^k']:.2f} |")
    L += ["", "## Per task", "", "| task | category | successes | failing checks |", "|---|---|---|---|"]
    for tid, d in s["per_task"].items():
        fc = ", ".join(f"{k}x{v}" for k, v in d["failed_checks"].items()) or "-"
        L.append(f"| {tid} | {d['category']} | {d['successes']}/{d['n']} | {fc} |")
    L += ["", "## Cost / behaviour", "",
          f"- avg tool calls per episode: {s['avg_tool_calls']:.2f}", f"- avg agent turns: {s['avg_agent_turns']:.2f}",
          f"- avg latency per episode: {s['avg_latency_s']:.1f}s",
          f"- tokens: prompt {s['tokens']['prompt']}, completion {s['tokens']['completion']}"]
    if s["judge_mean_score"] is not None:
        L.append(f"- LLM-judge mean rubric score (soft, uncalibrated): {s['judge_mean_score']:.2f}")
    L += ["", "Read the failing transcripts in `trials.jsonl` before trusting any number: a failure can be a bad agent *or* a bad grader.", ""]
    return "\n".join(L)
