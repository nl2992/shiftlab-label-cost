# Real-data inputs

The supported real datasets are defined in `configs/promising_directions/real_datasets.json`. The frozen model-pair loss files and their matching manifest are in `docs/promising_directions/results/b3_plus/all_streams/`.

Use `scripts/promising_directions/b3_plus_all_streams.py` to replay the primary policies and `paper/b3_plus/figures/make_all_streams.py` to regenerate all primary manuscript results. The replay uses the recorded loss arrays directly.
