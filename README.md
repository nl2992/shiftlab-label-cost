# What Does It Cost to Keep a Certificate? Settled-Decision Label Acquisition

Anytime-valid procedures certify a model swap, but they pay for the certificate in labels, and under a hard label budget that price is a first-order cost. A disagreement-gated paired e-process keeps buying labels at its budget-paced rate long after the deployment decision is settled. Settled-decision acquisition withholds queries in the two states where no label can change the deployed action, so it can only spend less. The repository holds the method, baselines, five real case studies selected from 36 screened datasets, ablations and the manuscript.

**Status.** Manuscript draft.

## Layout

- `paper/`: manuscript sources
- `docs/`: protocols, results and notes
- `src/shiftlab/`: implementation
- `scripts/`: experiment, analysis and audit scripts
- `tests/`: unit and regression tests
- `configs/`: experiment configuration
- `research/`: dataset discovery

The manuscript sources are in `paper/b3_plus`. The main implementation is `scripts/promising_directions`; scripts are under `scripts/promising_directions`. The dataset screening pipeline is in `research/b3_plus/dataset_discovery/`; results and protocols are in `docs/promising_directions/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Large raw data and run outputs are not tracked. Scripts regenerate them, and each protocol or results note under `docs/` states the seeds and commands it used.

## Related repositories

The Python package `shiftlab` started as one shared codebase and is now split by paper. Each repository carries the part of it that the paper needs.

## License

MIT. See `LICENSE`.
