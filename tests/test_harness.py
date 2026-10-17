"""Sanity tests for the harness itself (no model needed):  python -m tests.test_harness

An eval you haven't tested is a second unreliable system. These check that the graders agree with
agents whose behaviour we know by construction.
"""
import tempfile

from agenteval.agent import make_user, run_episode
from agenteval.graders import _extract_json, check_state, check_verify_before_write, grade, judge
from agenteval.llm import MockBackend, Reply
from agenteval.metrics import pass_at_k, pass_hat_k
from agenteval.runner import load_scenarios, run_suite

SC = load_scenarios("data/scenarios.json")


def run(mode, trials=6):
    with tempfile.TemporaryDirectory() as d:
        s, _ = run_suite(SC, MockBackend(mode), trials=trials, outdir=d, tag="t", progress=False)
    return s


def test_metrics():
    assert abs(pass_hat_k(4, 3, 2) - 0.5) < 1e-9      # C(3,2)/C(4,2)
    assert abs(pass_at_k(4, 1, 2) - 0.5) < 1e-9       # 1 - C(3,2)/C(4,2)
    assert pass_hat_k(1, 1, 2) is None


def test_good_agent_passes_everything():
    s = run("good", 3)
    assert s["overall"][1]["pass^k"] == 1.0, s["per_task"]


def test_sloppy_agent_is_caught():
    s = run("sloppy", 2)
    ok = [t for t, d in s["per_task"].items() if d["successes"] == d["n"]]
    assert ok == ["s12_off_topic"], ok  # only the task where doing nothing is right


def test_flaky_agent_shows_reliability_gap():
    s = run("flaky0.3", 8)
    o = s["overall"]
    assert o[8]["pass@k"] > o[1]["pass@k"] and o[8]["pass^k"] < o[1]["pass^k"] - 0.3, o


def test_state_check_detects_collateral_damage():
    from agenteval.env import ShopEnv
    env = ShopEnv()
    env.call("cancel_order", {"order_id": "O1002"})
    assert check_state({"state": {}}, env.db)            # unexpected change reported
    assert not check_state({"state": {"O1002": {"status": "cancelled"}}}, env.db)


def test_verify_before_write():
    ok = [{"name": "get_order", "args": {}, "result": {"order_id": "O1", "customer_id": "C1"}},
          {"name": "get_customer", "args": {}, "result": {"customer_id": "C1"}},
          {"name": "issue_refund", "args": {"order_id": "O1"}, "result": {}}]
    assert not check_verify_before_write(ok)
    wrong_owner = [ok[0], {"name": "get_customer", "args": {}, "result": {"customer_id": "C2"}}, ok[2]]
    assert check_verify_before_write(wrong_owner)
    assert check_verify_before_write([ok[2]])


def test_judge_parsing_and_unknown_handling():
    class Stub:
        name = "stub"

        def chat(self, *a, **k):
            return Reply(content='noise {"items":[{"item":1,"verdict":"pass"},{"item":2,"verdict":"unknown"},{"item":3,"verdict":"fail"}]} tail')

    sc = {"rubric": ["a", "b", "c"], "expect": {}}
    ep = run_episode(SC[0], MockBackend("good"), make_user(SC[0]))
    j = judge(sc, ep, Stub())
    assert j["score"] == 0.5 and j["unknown"] == 1
    assert _extract_json('x {"a": 1} y') == {"a": 1}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
