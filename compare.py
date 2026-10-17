#!/usr/bin/env python3
"""Regression check between two runs:  python compare.py results/<baseline> results/<candidate>

Prints per-task success deltas. Exit code 1 if any task got worse (use it in CI after a prompt/model change).
With few trials, treat 1-trial wiggles as noise: raise --trials before drawing conclusions.
"""
import json
import sys


def load(d):
    return json.load(open(f"{d}/summary.json"))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    worse = 0
    print(f"{'task':45s} {'base':>8s} {'new':>8s}")
    for tid in sorted(set(a["per_task"]) | set(b["per_task"])):
        x, y = a["per_task"].get(tid), b["per_task"].get(tid)
        fx = f"{x['successes']}/{x['n']}" if x else "-"
        fy = f"{y['successes']}/{y['n']}" if y else "-"
        flag = ""
        if x and y and y["successes"] / y["n"] < x["successes"] / x["n"] - 1e-9:
            flag, worse = "  <-- REGRESSION", worse + 1
        print(f"{tid:45s} {fx:>8s} {fy:>8s}{flag}")
    print(f"\npass^1: {a['overall']['1']['pass^k']:.2f} -> {b['overall']['1']['pass^k']:.2f}")
    sys.exit(1 if worse else 0)


if __name__ == "__main__":
    main()
