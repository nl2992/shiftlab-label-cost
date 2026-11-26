"""Authoritative real-dataset scope and integrity checks for the manuscript."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/promising_directions/real_datasets.json"
DATASETS = json.loads(CONFIG.read_text())["datasets"]
STREAMS = tuple(DATASETS)
DELAYS = {name: meta["delay"] for name, meta in DATASETS.items()}
OUT = ROOT / "docs/promising_directions/results/b3_plus/all_streams"
LOSSES = OUT / "losses"


def require_scope(names):
    actual, expected = set(names), set(STREAMS)
    if actual != expected:
        raise ValueError(f"dataset scope mismatch: missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")


def validate_losses():
    require_scope(p.stem for p in LOSSES.glob("*.npz"))
    for name, meta in DATASETS.items():
        path = LOSSES / f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"loss checksum mismatch: {name}")


def load_results():
    manifest = json.loads((OUT / "manifest.json").read_text())
    require_scope(manifest)
    if manifest != DATASETS:
        raise ValueError("manifest differs from authoritative dataset configuration")
    results = json.loads((OUT / "results.json").read_text())
    require_scope(results)
    for name, result in results.items():
        if result["delay"] != DELAYS[name] or result["K"] != len(result["episodes"]):
            raise ValueError(f"invalid episode metadata: {name}")
    return {name: results[name] for name in STREAMS}
