"""P0, P3b, P6 and P7s on the 20 configured real loss streams, primary configuration.

T=2000, B=200, c_q=0.05, c_s=5, 100 query seeds per episode, seeded as in the case studies
(``episode_u``). Every configured stream is run and reported.

    python scripts/promising_directions/b3_plus_all_streams.py --collect <dir> [<dir> ...]
    python scripts/promising_directions/b3_plus_all_streams.py

``--collect`` copies the cached 0/1 loss files and their delays next to the results, once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b3_plus_experiments import ROOT, b1, episode_u, run_variants, summarise_variant
from real_datasets import DATASETS, require_scope, validate_losses

OUT = ROOT / "docs/promising_directions/results/b3_plus/all_streams"
LOSSES = OUT / "losses"
VARIANTS = ("P0", "P3b", "P6", "P7s")
N_SEEDS, BUDGET, CS, CQ = 100, 200, 5.0, 0.05
def collect(dirs: list[Path]) -> None:
    """Collect exactly the configured loss files, validating all inputs before copying."""
    sources = {}
    for directory in dirs:
        for path in sorted(directory.glob("*.npz")):
            name = path.stem
            if name not in DATASETS or name in sources:
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != DATASETS[name]["sha256"]:
                raise ValueError(f"cache differs from configured losses: {name}")
            sources[name] = path
    require_scope(sources)
    LOSSES.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination = LOSSES / f"{name}.npz"
        if source.resolve() != destination.resolve():
            shutil.copy(source, destination)
    (OUT / "manifest.json").write_text(json.dumps(DATASETS, indent=1) + "\n")
    validate_losses()
    print(f"collected {len(sources)} streams")


def run_stream(item):
    name, meta = item
    z = np.load(LOSSES / f"{name}.npz")
    p = b1.Params(T=2000, delay=meta["delay"], cq=CQ)
    episodes = []
    for ep, (l0, l1) in enumerate(zip(z["l0"], z["l1"])):
        u = episode_u(ep, N_SEEDS, len(l0))
        res = run_variants(l0, l1, u, BUDGET, CS, CQ, p, VARIANTS)
        row = {}
        for v in VARIANTS:
            s = summarise_variant(res[v], CQ, CS)
            s.pop("cost_seeds")
            row[v] = s
        a, b = res["P3b"], res["P7s"]
        # Matching deployment summaries on every seed: loss, switches and rollbacks.
        row["same_deployment"] = all(
            np.array_equal(a[k], b[k]) for k in ("loss", "switches", "rollbacks")
        )
        episodes.append(row)
    return name, {"delay": meta["delay"], "K": len(episodes), "episodes": episodes}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", nargs="+", type=Path)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args(argv)
    if a.collect:
        collect(a.collect)
        return
    manifest = json.loads((OUT / "manifest.json").read_text())
    require_scope(manifest)
    if manifest != DATASETS:
        raise ValueError("manifest differs from authoritative dataset configuration")
    validate_losses()
    with Pool(a.workers) as pool:
        results = dict(pool.map(run_stream, sorted(manifest.items()), chunksize=1))
    (OUT / "results.json").write_text(json.dumps(results) + "\n")
    print(f"ran {len(results)} streams, {sum(r['K'] for r in results.values())} episodes")


if __name__ == "__main__":
    main()
