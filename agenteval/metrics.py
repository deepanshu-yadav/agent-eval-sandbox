"""pass@k / pass^k (tau-bench). With n trials and c successes on one task:

  pass@k = 1 - C(n-c, k) / C(n, k)   probability that at least one of k attempts succeeds (retry-able tools)
  pass^k =     C(c,   k) / C(n, k)   probability that ALL k attempts succeed (consistency users feel)
"""
from __future__ import annotations

from math import comb


def pass_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n, c, k):
    if n < k:
        return None
    return comb(c, k) / comb(n, k)


def aggregate(per_task: dict[str, tuple[int, int]], ks):
    """per_task: id -> (n, c). Returns {k: {"pass@k": x, "pass^k": y}} averaged over tasks."""
    out = {}
    for k in ks:
        a = [pass_at_k(n, c, k) for n, c in per_task.values() if n >= k]
        h = [pass_hat_k(n, c, k) for n, c in per_task.values() if n >= k]
        if a:
            out[k] = {"pass@k": sum(a) / len(a), "pass^k": sum(h) / len(h)}
    return out
