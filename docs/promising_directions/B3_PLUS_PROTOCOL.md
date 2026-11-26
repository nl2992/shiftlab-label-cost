# Real-data evaluation protocol

The real-data scope is exactly the 20 IDs in `configs/promising_directions/real_datasets.json`. That configuration records each fixed label delay, original loss-cache source and SHA-256 hash. `real_datasets.py` rejects missing or additional streams and checks every loss file against its recorded hash before regeneration.

Primary configuration: 2,000 events per episode, budget 200 labels, query price 0.05, switch cost 5, and 100 paired query seeds. The seed base is 20261101; query draws depend only on episode and seed index. All policies use the same loss episodes and draws. Delays are 24, 50, 96 or 100 events as recorded per stream.

The primary policies are P0 (never switch), P3b (the pinned certified switching procedure), P6 (fixed-sample paired binomial test), and P7s (settled-decision acquisition). Policy means first average seeds within each episode and then episodes within a stream. Cohort means of percentage savings and medians weight streams equally. Cost is deployment loss plus query price times labels plus switch cost times switches. Label savings and cost savings are computed before display rounding.

Uncertainty intervals in the manuscript appendix use 10,000 paired episode-bootstrap resamples for streams with at least five episodes. Shorter streams have no episode interval. Randomisation seeds are not treated as independent data episodes. Electricity meters and air-quality stations may be correlated.

The saved `same_deployment` flag compares deployment loss, switch count and rollback count for every query seed within an episode. Equal totals do not prove identical event-by-event deployment trajectories.

Ablation records are available for spam and rialto. Sensitivity records cover those streams and openml_ldpa: S0 primary, S1 budget 400, S2 switch cost 20, S3 doubled label delay, S4 query price 0.10. A configuration without a recorded run is shown as unavailable. No sensitivity or ablation result is extrapolated to the entire cohort.

Rebuild derived artifacts with `python paper/b3_plus/figures/make_all_streams.py`. Replay all primary policies with `python scripts/promising_directions/b3_plus_all_streams.py --workers 7`. Frozen model-pair losses are the replay inputs; raw-data downloads and model retraining are outside this replay pipeline.
