#!/usr/bin/env python3
"""Print failing trials with the exact reason and the full transcript.

  python show_failures.py                      # latest run in ./results
  python show_failures.py results\\20261017-130000 --task s01 --limit 2
"""
import argparse
import glob
import json
import os
import sys

from agenteval.agent import render_transcript


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="run folder (default: newest under results/)")
    ap.add_argument("--task", help="substring of scenario id")
    ap.add_argument("--limit", type=int, default=3)
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows redirects/pipes default to cp1252; model output may not be ASCII
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    run = a.run or max(glob.glob("results/*/"), key=os.path.getmtime)
    shown = 0
    for line in open(os.path.join(run, "trials.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["grade"]["success"] or (a.task and a.task not in r["scenario_id"]):
            continue
        print("=" * 90)
        print(f"{r['scenario_id']} trial {r['trial']}  stop_reason={r['episode']['stop_reason']}")
        for name in r["grade"]["failed"]:
            print(f"  FAILED {name}: {r['grade']['checks'][name]['detail']}")
        if r["episode"].get("error"):
            print("  ERROR:", r["episode"]["error"])
        print("-" * 90)
        print(render_transcript(r["episode"]))
        shown += 1
        if shown >= a.limit:
            break
    if not shown:
        print("no failing trials matched")


if __name__ == "__main__":
    main()
