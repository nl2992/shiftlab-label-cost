# What Does It Cost to Keep a Certificate? Settled-Decision Label Acquisition

Settled-decision acquisition reduces label spending by attenuating queries after a deployment decision settles. This repository evaluates the method on **20 real datasets and 307 episodes**, with frozen model-pair losses and 100 paired query seeds per episode.

Across the 20 datasets, mean savings relative to P3b are **1.43% in net cost and 37% in labels**; the corresponding medians are **0.95% and 30%**. Each dataset has equal weight.

The authoritative dataset list, delays and loss checksums are in `configs/promising_directions/real_datasets.json`. Primary results and frozen replay inputs are in `docs/promising_directions/results/b3_plus/all_streams/`. The manuscript is in `paper/b3_plus/`, and the implementation and tests are in `scripts/promising_directions/` and `tests/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Reproduce results

Rebuild the manuscript tables, figures and results report from saved episode records:

```bash
python paper/b3_plus/figures/make_all_streams.py
```

Rerun all primary policies on the 20 checksum-verified loss caches:

```bash
python scripts/promising_directions/b3_plus_all_streams.py --workers 7
python paper/b3_plus/figures/make_all_streams.py
```

See `docs/promising_directions/B3_PLUS_PROTOCOL.md` for configuration and aggregation, `docs/promising_directions/B3_PLUS_RESULTS.md` for results, and `paper/b3_plus/BUILD.md` to build the paper. Detailed ablation records cover two retained streams; sensitivity records cover three. Synthetic theory experiments remain separate from the real-data evaluation.

## License

MIT. See `LICENSE`.
