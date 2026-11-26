"""Pinned certified-switching and fixed-sample baselines for frozen real loss replay."""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from functools import cache
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
B1_PATH = ROOT / "scripts/promising_directions/b1_certified_switching.py"
B1_COMMIT = "041bc26"
B1_SHA256 = "7565a647f3bb3f940d4fbd2a948597b9ffe42ef0485550a771d137e7930baebd"

T_EP = 2000
N_SEEDS = 100
SEED_BASE = 20261101
BOOT = 2000
CQ = 0.05
SAFE_MARGIN = 0.02
P6_N = 50
POLICIES = ("P0", "P1", "P2", "P3b", "P6")
SETTINGS = {
    "B200_cs5": (200, 5.0),
    "B200_cs20": (200, 20.0),
    "B400_cs5": (400, 5.0),
}
PRIMARY = "B200_cs5"

# Configured delays for the retained real datasets.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_datasets import DELAYS
STREAMS = {name: {"delay": delay} for name, delay in DELAYS.items()}


# ----------------------------------------------------------------------------- b1 import
def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
    )


def load_b1():
    """Import b1_certified_switching as at commit 041bc26. Returns (module, record)."""
    rel = str(B1_PATH.relative_to(ROOT)).replace("\\", "/")
    unchanged = hashlib.sha256(B1_PATH.read_bytes()).hexdigest() == B1_SHA256
    path, how = B1_PATH, "pinned file (sha256 matches the version used at 041bc26)"
    if not unchanged:
        raise RuntimeError(f"{rel} differs from the pinned version (sha256 {B1_SHA256[:12]}...)")
    spec = importlib.util.spec_from_file_location("b1_certified_switching", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    rec = {"commit": B1_COMMIT, "source": how, "path": str(path)}
    rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return mod, rec


b1, B1_RECORD = load_b1()


# ----------------------------------------------------------------------------- episodes
def episode_plan(start: int, length: int, kmax: int = 50) -> list[int]:
    """K = min(kmax, floor(L / 2000)) non-overlapping episodes at start + i * floor(L / K)."""
    k = min(kmax, length // T_EP)
    step = length // k
    return [start + i * step for i in range(k)]


def episode_u(ep_index: int, n_seeds: int) -> np.ndarray:
    """(T, n_seeds) query-randomisation draws; column j depends only on (episode, seed j)."""
    cols = [
        np.random.default_rng(np.random.SeedSequence([SEED_BASE, ep_index, j])).random(T_EP)
        for j in range(n_seeds)
    ]
    return np.stack(cols, axis=1)


# ----------------------------------------------------------------------------- P6
@cache
def p6_reject(k: int, n: int) -> bool:
    """One-sided exact binomial test of H0: P(challenger correct | disagreement) <= 1/2."""
    if n <= 0:
        return False
    return bool(binomtest(int(k), int(n), 0.5, alternative="greater").pvalue <= 0.05)


def run_p6(l0, l1, u, budget: int, p) -> dict:
    """P6: query disagreements with P3b's pacing; test the first 50 released labels once.

    Labels are issued at t and released at t + delay (as in b1's Cert). Issuing stops once the test
    has run. If the budget is exhausted before 50 labels exist, the test runs on the released
    labels once all issued ones are released. Promote on rejection; never roll back.
    """
    t_len, n = u.shape
    dis_all = l0 != l1
    dep = np.zeros(n, dtype=np.int8)
    loss = np.zeros(n)
    rem = np.full(n, budget)
    queries = np.zeros(n, dtype=np.int64)
    xbuf = np.zeros((p.delay, n), dtype=np.int8)  # 0 none, 1 challenger correct, 2 incumbent
    rel = np.zeros(n, dtype=np.int64)
    kcnt = np.zeros(n, dtype=np.int64)
    tested = np.zeros(n, dtype=bool)
    t_prom = np.full(n, np.nan)
    ndis = np.zeros(n)
    for t in range(t_len):
        l0t, l1t = l0[t], l1[t]
        loss += np.where(dep == 0, l0t, l1t)
        slot = t % p.delay
        rate = (ndis + p.prior_rate * p.prior_w) / (t + p.prior_w)
        q = np.clip(rem / np.maximum(rate * (p.T - t), 1e-9), p.pmin, 1.0)
        q = np.where(rem > 0, q, 0.0)
        ask = dis_all[t] & (u[t] < q) & ~tested
        rem = rem - ask
        queries += ask
        if t - p.delay >= 0:
            code = xbuf[slot]
            got = code > 0
            take = got & (rel < P6_N)
            kcnt += take & (code == 1)
            rel += got
        xbuf[slot] = np.where(ask, np.where(l1t == 0, 1, 2), 0).astype(np.int8)
        in_flight = queries - rel
        fire = ~tested & ((rel >= P6_N) | ((rem == 0) & (in_flight == 0) & (queries > 0)))
        for i in np.flatnonzero(fire):
            if p6_reject(int(kcnt[i]), int(min(rel[i], P6_N))):
                dep[i] = 1
                t_prom[i] = t + 1
        tested |= fire
        ndis = ndis + dis_all[t]
    return {
        "loss": loss,
        "queries": queries,
        "switches": (~np.isnan(t_prom)).astype(np.int64),
        "rollbacks": np.zeros(n, dtype=np.int64),
        "t_prom": t_prom,
    }


# ----------------------------------------------------------------------------- one episode
def run_episode(l0, l1, u, budget: int, delay: int, base_err: float) -> dict:
    """All five policies on one episode. l0, l1: (T,) 0/1 losses; u: (T, n_seeds)."""
    p = b1.Params(T=T_EP, delay=delay, base_err=base_err, eps=0.05)
    n = u.shape[1]
    a0 = np.repeat(np.asarray(l0, dtype=np.int8)[:, None], n, axis=1)
    a1 = np.repeat(np.asarray(l1, dtype=np.int8)[:, None], n, axis=1)
    res = b1.run_policies(a0, a1, u, budget, (), p, b2=True)
    p0, p1 = b1.simple_out(a0, a1)
    return {
        "P0": p0,
        "P1": p1,
        "P2": res["P2"],
        "P3b": res["P3b"],
        "P6": run_p6(a0, a1, u, budget, p),
    }


def episode_rows(stream: str, ep: int, start: int, pols: dict, delay: int, n_seeds: int, st):
    rows = []
    for key, (budget, cs) in SETTINGS.items():
        out = {}
        for name in POLICIES:
            d = pols[budget][name]
            cost = d["loss"] + CQ * d["queries"] + cs * d["switches"]
            out[name] = {
                "mean_cost": float(cost.mean()),
                "mean_loss": float(np.mean(d["loss"])),
                "mean_labels": float(np.mean(d["queries"])),
                "mean_switches": float(np.mean(d["switches"])),
                "mean_rollbacks": float(np.mean(d["rollbacks"])),
                "promotion_rate": float(np.mean(~np.isnan(d["t_prom"]))),
                "cost_seeds": [round(float(x), 6) for x in cost],
            }
        rows.append(
            {
                "stream": stream,
                "episode": ep,
                "start": int(start),
                "setting": key,
                "budget": budget,
                "cs": cs,
                "delay": delay,
                "n_seeds": n_seeds,
                "episode_stats": st,
                "policies": out,
            }
        )
    return rows


def boot_diff(per_ep: np.ndarray, per_seed: np.ndarray, rng) -> dict:
    """CI for the mean of a paired difference: episodes if K >= 5, else seeds (LOW-POWER)."""
    low = len(per_ep) < 5
    arr = per_seed if low else per_ep
    idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
    m = arr[idx].mean(1)
    return {
        "mean": float(per_ep.mean()),
        "lo": float(np.percentile(m, 2.5)),
        "hi": float(np.percentile(m, 97.5)),
        "bootstrap_level": "seed" if low else "episode",
        "low_power": bool(low),
    }


def stream_summary(rows: list[dict], setting: str, rng) -> dict:
    rs = sorted((r for r in rows if r["setting"] == setting), key=lambda r: r["episode"])
    ep = {p: np.array([r["policies"][p]["mean_cost"] for r in rs]) for p in POLICIES}
    seeds = {p: np.mean([r["policies"][p]["cost_seeds"] for r in rs], axis=0) for p in POLICIES}
    mean = {p: float(ep[p].mean()) for p in POLICIES}

    def diff(a, b):
        return boot_diff(ep[a] - ep[b], seeds[a] - seeds[b], rng)

    def avg(field, pol):
        return float(np.mean([r["policies"][pol][field] for r in rs]))

    denom = mean["P0"] - min(mean["P0"], mean["P1"])
    spend = CQ * avg("mean_labels", "P3b")
    bound = mean["P0"] + spend + SAFE_MARGIN * mean["P0"]
    d_p0 = diff("P3b", "P0")
    return {
        "episodes": len(rs),
        "mean_cost": mean,
        "paired": {
            "P3b_minus_P0": d_p0,
            "P3b_minus_P2": diff("P3b", "P2"),
            "P3b_minus_P1": diff("P3b", "P1"),
            "P3b_minus_P6": diff("P3b", "P6"),
        },
        "headroom_capture": float((mean["P0"] - mean["P3b"]) / denom) if denom > 0 else None,
        "p3b_query_spend": spend,
        "safe_bound": bound,
        "WIN": bool(d_p0["hi"] < 0 and mean["P3b"] <= mean["P2"]),
        "SAFE": bool(mean["P3b"] <= bound),
        "diagnostics": {
            p: {
                "mean_labels": avg("mean_labels", p),
                "mean_switches": avg("mean_switches", p),
                "mean_rollbacks": avg("mean_rollbacks", p),
                "promotion_rate": avg("promotion_rate", p),
            }
            for p in POLICIES
        },
    }


def programme(streams: dict) -> dict:
    wins = [s for s, v in streams.items() if v["primary"].get("WIN")]
    safe_fail = [s for s, v in streams.items() if not v["primary"].get("SAFE")]
    if len(wins) >= 3 and not safe_fail:
        outcome = "SUCCESS"
    elif 1 <= len(wins) <= 2 or len(safe_fail) == 1:
        outcome = "PARTIAL"
    else:
        outcome = "FAIL"
    return {"outcome": outcome, "win_streams": wins, "safe_failures": safe_fail}


def locked_plan(stream: str) -> list[int]:
    """Read recorded episode offsets for retained legacy baseline comparisons."""
    if stream not in STREAMS:
        raise ValueError(f"unknown configured stream: {stream}")
    with np.load(ROOT / "docs/promising_directions/results/b3/losses" / f"{stream}.npz") as z:
        return z["starts"].tolist()
