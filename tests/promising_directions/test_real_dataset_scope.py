"""Dataset identity, numerical reconciliation and replay checks for manuscript results."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/promising_directions"))
import real_datasets as scope

EXPECTED = {
    "rialto", "metro", "spam", "air_quality_co", "bike_hour", "elec_MT_010",
    "air_Tiantan", "air_Dingling", "elec_MT_022", "air_Dongsi", "elec_MT_002",
    "openml_ldpa", "air_Wanliu", "air_Shunyi", "elec_MT_011", "elec_MT_014",
    "elec_MT_043", "elec_MT_027", "elec_MT_009", "elec_MT_041",
}


def test_exact_dataset_identity_and_loss_checksums():
    assert set(scope.STREAMS) == EXPECTED
    scope.validate_losses()
    results = scope.load_results()
    assert sum(r["K"] for r in results.values()) == 307
    for name, record in results.items():
        with np.load(scope.LOSSES / f"{name}.npz") as z:
            assert z["l0"].shape == z["l1"].shape == (record["K"], 2000)
            assert np.isin(z["l0"], [0, 1]).all()
            assert np.isin(z["l1"], [0, 1]).all()


def test_scope_rejects_missing_and_extra_streams():
    with pytest.raises(ValueError, match="scope mismatch"):
        scope.require_scope(EXPECTED - {"rialto"})
    with pytest.raises(ValueError, match="scope mismatch"):
        scope.require_scope(EXPECTED | {"outside_scope"})


def test_saved_costs_reconcile_to_components():
    for record in scope.load_results().values():
        for episode in record["episodes"]:
            for variant in ("P0", "P3b", "P6", "P7s"):
                r = episode[variant]
                assert r["mean_cost"] == pytest.approx(
                    r["mean_loss"] + .05*r["mean_labels"] + 5*r["mean_switches"], abs=1e-9)


def test_manuscript_summary_reconciles_to_all_episode_records():
    results = scope.load_results()
    summary = json.loads((scope.OUT / "summary.json").read_text())
    assert {r["name"] for r in summary["streams"]} == EXPECTED
    ratios, labels, cheaper, same = [], [], 0, 0
    for name, result in results.items():
        avg = lambda v, f: np.mean([e[v][f] for e in result["episodes"]])
        diff = avg("P7s", "mean_cost") - avg("P3b", "mean_cost")
        ratios.append(-100*diff/avg("P3b", "mean_cost"))
        labels.append(100*(1-avg("P7s", "mean_labels")/avg("P3b", "mean_labels")))
        cheaper += int(diff < -1e-9)
        same += sum(e["same_deployment"] for e in result["episodes"])
    n = summary["numbers"]
    assert n["NCheaper"] == cheaper == 19
    assert n["NDearer"] == 1
    assert n["NSameEpisodes"] == same == 275
    assert n["MeanCostChange"] == f"{np.mean(ratios):.2f}" == "1.43"
    assert n["MeanSaving"] == f"{np.mean(labels):.0f}" == "37"
    assert n["MedCostChange"] == f"{np.median(ratios):.2f}"
    assert n["MedSaving"] == f"{np.median(labels):.0f}"
    macros = (ROOT / "paper/b3_plus/tables/numbers_all.tex").read_text()
    for key, value in n.items():
        assert f"\\newcommand{{\\{key}}}{{{value}}}" in macros


def test_replay_matches_saved_episode_with_changed_deployment():
    from b3_plus_experiments import b1, episode_u, run_variants, summarise_variant
    name = "elec_MT_041"
    saved = scope.load_results()[name]["episodes"][0]
    with np.load(scope.LOSSES / f"{name}.npz") as z:
        result = run_variants(z["l0"][0], z["l1"][0], episode_u(0,100,2000),
                              200,5,.05,b1.Params(T=2000,delay=scope.DELAYS[name],cq=.05),
                              ("P0","P3b","P6","P7s"))
    for variant, raw in result.items():
        got = summarise_variant(raw,.05,5)
        for field in ("mean_cost", "mean_loss", "mean_labels", "mean_switches", "mean_rollbacks"):
            assert got[field] == pytest.approx(saved[variant][field], abs=1e-9)
