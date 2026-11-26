# Real-data results

The evaluation contains 20 real datasets and 307 episodes. Each episode has 2,000 events and 100 paired query seeds; B=200, query cost=0.05, switch cost=5.

Ours is cheaper than P3b on 19 streams, equal on 0, and dearer on 1. Mean net-cost saving: 1.43%; mean labels saved: 37%. The corresponding medians are 0.95% and 30%. P6 is cheaper than ours on 5 streams.

Deployment loss, switch count and rollback count match P3b on every query seed in 275 of 307 episodes. Equal totals do not imply identical action trajectories.

Stream summaries weight episodes equally; cohort medians weight streams equally. Bootstrap intervals use 10,000 paired episode resamples for streams with at least five episodes; they are omitted for shorter streams. Related meters and stations are correlated.

| Stream | Episodes | Net cost saved | Labels | Labels saved |
|---|---:|---:|---:|---:|
| rialto | 16 | +5.12% | 159 → 53 | 67% |
| metro | 16 | +4.45% | 200 → 64 | 68% |
| spam | 1 | +2.62% | 200 → 67 | 67% |
| air_quality_co | 2 | +2.07% | 178 → 45 | 75% |
| bike_hour | 6 | +1.88% | 198 → 89 | 55% |
| elec_MT_010 | 20 | +1.23% | 162 → 105 | 35% |
| air_Tiantan | 12 | +1.19% | 173 → 138 | 20% |
| air_Dingling | 12 | +1.09% | 154 → 108 | 30% |
| elec_MT_022 | 20 | +0.98% | 144 → 93 | 35% |
| air_Dongsi | 12 | +0.98% | 173 → 118 | 32% |
| elec_MT_002 | 20 | +0.93% | 122 → 89 | 27% |
| openml_ldpa | 32 | +0.92% | 181 → 115 | 36% |
| air_Wanliu | 12 | +0.88% | 187 → 154 | 17% |
| air_Shunyi | 11 | +0.77% | 169 → 119 | 29% |
| elec_MT_011 | 20 | +0.76% | 161 → 114 | 29% |
| elec_MT_014 | 20 | +0.73% | 153 → 115 | 25% |
| elec_MT_043 | 20 | +0.71% | 123 → 93 | 25% |
| elec_MT_027 | 20 | +0.68% | 159 → 111 | 30% |
| elec_MT_009 | 20 | +0.67% | 185 → 131 | 29% |
| elec_MT_041 | 15 | -0.09% | 147 → 134 | 9% |

Regenerate with `python paper/b3_plus/figures/make_all_streams.py`. The authoritative scope is `configs/promising_directions/real_datasets.json`; frozen loss files and their SHA-256 hashes are under `results/b3_plus/all_streams/`. Ablation measurements cover spam and rialto; sensitivity measurements cover these and openml_ldpa.
