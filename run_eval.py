#!/usr/bin/env python3
"""Run the ShopDesk agent eval.

Examples
  # OpenAI-compatible API key
  export OPENAI_API_KEY=sk-...
  python run_eval.py --agent openai:gpt-4.1-mini --trials 5

  # Anthropic API key
  export ANTHROPIC_API_KEY=...
  python run_eval.py --agent anthropic:<model-id> --trials 5

  # Fully local with llama.cpp (start:  llama-server -m model.gguf --jinja -c 16384 -ngl 99 --port 8080)
  python run_eval.py --agent llamacpp:local@http://localhost:8080/v1 --trials 5 --workers 1

  # LLM-simulated user + LLM judge (can be a different / stronger model than the agent)
  python run_eval.py --agent llamacpp:local --user openai:gpt-4.1-mini --judge openai:gpt-4.1-mini

  # No model at all: validate graders/metrics with the rule-based fake agent
  python run_eval.py --agent mock:flaky0.3 --trials 8
"""
import argparse
import json
import sys

from agenteval.llm import parse_spec
from agenteval.runner import load_scenarios, render_report, run_suite


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="spec of the agent under test, e.g. openai:gpt-4.1-mini")
    ap.add_argument("--user", help="spec for an LLM user simulator (default: scripted users, fully deterministic)")
    ap.add_argument("--judge", help="spec for the optional LLM judge (soft rubric score)")
    ap.add_argument("--scenarios", default="data/scenarios.json")
    ap.add_argument("--only", nargs="*", help="substring filter on scenario ids")
    ap.add_argument("--category", nargs="*")
    ap.add_argument("--trials", type=int, default=3, help="k independent trials per task (needed for pass^k)")
    ap.add_argument("--workers", type=int, default=4, help="parallel episodes (use 1 for a single local llama.cpp slot)")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0, help="use >0 (e.g. 0.7) so trials differ")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", help="run folder name (default: timestamp)")
    ap.add_argument("--fail-under", type=float, help="CI gate: exit 1 if pass^1 is below this")
    ap.add_argument("--api-key", help="API key for the model (or pass via environment variable or #KEY in spec)")
    ap.add_argument("--live", action="store_true", default=True, help="stream live logs for turns and tool calls (default: True)")
    ap.add_argument("--no-live", dest="live", action="store_false", help="disable live logs, only print trial completions")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows redirects/pipes default to cp1252; model output may not be ASCII
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    agent = parse_spec(a.agent, api_key=a.api_key)
    user = parse_spec(a.user) if a.user else None
    jdg = parse_spec(a.judge) if a.judge else None
    sc = load_scenarios(a.scenarios, a.only, a.category)
    if not sc:
        sys.exit("no scenarios selected")
    summary, outdir = run_suite(sc, agent, a.trials, user, jdg, a.workers, a.max_turns, a.temperature, a.out, a.tag, live=a.live)
    if summary:
        print("\n" + render_report(summary))
        print(f"Saved to {outdir}/ (trials.jsonl has full transcripts)")
        if a.fail_under is not None and summary["overall"][1]["pass^k"] < a.fail_under:
            sys.exit(1)
    else:
        print(f"\nNo complete summary generated (interrupted). Saved partial trials to {outdir}/")


if __name__ == "__main__":
    main()
