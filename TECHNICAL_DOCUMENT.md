# BarcelonAI technical details

More than a Hack 2027, Phase 1. The [four-page report](output/pdf/BarcelonAI.pdf) explains the task and results in plain language. This file describes the implementation and reproduction.

Team: Sayfullah Jumoorty; Hamdullah Dadabhoy; Muaaz Bassa; Rayhaan Hanslod; Muzzimmil Sakidas.

## Two methods

**Method 1 (v9)** combines ball-to-player proximity, team shape, player movement, algorithm-generated shape labels, a Hidden Markov Model, gap-mixture/liveness estimates, kickoff/restart detection and automatic boundary adjustment. It can borrow broadcast ball observations after automatic alignment checks; tactical observations remain the basis for liveness. It learns from algorithm-generated examples without human labels from the competition match. The external adapter retains the method but disables broadcast borrowing because external matches have no second-camera asset.

**Method 2 (final submission)** is a histogram gradient-boosted classifier followed by Viterbi decoding. It learns from labelled external matches. Its 120 features comprise geometry, motion, ball proximity, gap duration, previous/next credible control and temporal context. It uses tactical tracking only. It does not blend Method 1 predictions; an earlier prototype did. The method is offline and uses future observations.

EDA found the ball missing from 40.7% of graded frames, with 99.7% of missing time occurring in gaps over five seconds. This motivates training with stretches of absent observations. Automatic tactical tags were explored but excluded from the classifier. Neither video nor homography is used in final model fitting or inference. No manual corrections or match-frame-specific overrides are applied.

## External data and split

Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*. [Paper](https://doi.org/10.1038/s41597-025-04505-y), [dataset](https://doi.org/10.6084/m9.figshare.28196177), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

These are other German league matches. Targets use the provider's existing BallStatus and BallPossession fields: inactive maps to D; active home/away ownership maps to H/A. Unknown ownership is ungraded. Our team did not annotate these matches. We parse the XML, normalize centred metre coordinates and km/h speeds, and add a separating ungraded frame between halves.

| Role | Match IDs |
|---|---|
| Training | J03WPY, J03WN1, J03WMX, J03WOH |
| Development | J03WQQ |
| Held-out final tests | J03WOY, J03WR9 |
| Competition match | Unlabelled inference; public sample only for post-selection sanity checking |

The final split and promotion rule are recorded in `cook/reports/v2/protocol.md`. The older prototype used two training matches and a different development/test split. Its original test becomes part of final training, so its original test score is not independent evidence about Method 2. All final comparisons use the two new held-out matches.

## Features, corruption and fitting

Corruption alternates visible/hidden ball runs drawn from durations measured in unlabelled competition tracking. It adds temporally correlated spatial noise and removes random player tracks in two-second blocks. Corruption never reads possession targets. Seeds 2027 and 2028 provide two training corruptions; evaluation uses the fixed default corruption. This approximates sensor failures but does not duplicate actual camera-dependent missingness.

Credible control requires a visible ball within 2 m of a player, a nearest-opponent distance margin above 1.5 m, ball speed below 12 m/s and the ball more than 0.3 m inside the pitch. Previous/next controlling team, ages and agreement become features, not hard overrides. Features summarize teams without relying on persistent player identity.

Method 2 adds short offsets of ±5 and ±12 frames to the earlier 100-feature design. Training samples every fifth frame and exchanges home/away teams, producing 456,772 correlated training rows. The classifier uses 300 iterations, learning rate 0.06, maximum 15 leaves, minimum 60 samples per leaf, L2 regularization 8 and no random early-stopping split. Predictions at 5 Hz are interpolated to 25 Hz. Transition counts are learned from training labels and symmetrized between H/A. Each half is decoded separately; X marks outside-half frames.

The development search tried three classifiers (15 leaves, 31 leaves and 31 leaves with short-spell weighting), three mixture weights and four evidence scales: 36 configurations. Highest development spellF1, with macroF1 as a tie-break, selected 15 leaves, no Method 1 mixture and evidence scale 1. The final candidate was frozen before either test was scored. It was promoted only after beating the frozen earlier prototype on both tests. Test results did not refit model weights. The public sanity check did not change the recommendation.

## Results and interpretation

The report contains all four official published private baselines and a separate table recalculating those baselines on the public sample. Private scores and public scores must not be directly compared. Our private score is unknown.

| Same-input evaluation | Official smooth50 spellF1 | Method 1 spellF1 | Method 2 spellF1 |
|---|---:|---:|---:|
| External J03WOY | 0.1812 | 0.2595 | 0.5963 |
| External J03WR9 | 0.1381 | 0.1973 | 0.5917 |
| Public five-minute sanity check | 0.3256 | 0.4353 | 0.5622 |

Method 2's two-match mean spellF1 is 0.5940 and mean macroF1 is 0.9019. Method 1's corresponding means are 0.2284 and 0.5910. The earlier learned prototype reached mean spellF1 0.5477 and macroF1 0.8882 on these same tests. All scores use the unchanged official scorer. External baseline algorithms and both methods receive the same degraded tracking.

The public sample contains only 17 true possessions. Bootstrap intervals overlap; a small public gain is not sufficient evidence of generalisation. The final model also undercounts external possessions: 353 versus 456 and 304 versus 370. Short control changes and ambiguous stoppages remain difficult. Two test matches cannot establish performance across all clubs, leagues and providers; clubs can overlap between different matches.

## Generate the submission

Use Python 3.13 and the pinned requirements. Supply the licensed tactical ATD assets using the layout in `DATA.md`. No external downloads are needed for inference from the included model.

```bash
python3.13 -m venv .venv-barcelonai
.venv-barcelonai/bin/python -m pip install -r requirements-cook.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m cook.submit
.venv-barcelonai/bin/python evaluation/validate_submission.py evaluation/results/BarcelonAI.csv
```

Final model: `models/v2_leaf15.joblib`, SHA-256 `5268d8d413db5cf6faa056171847056c7f719a75296b95a0a412cb425cb986a0`.

Final CSV: `evaluation/results/BarcelonAI.csv`, SHA-256 `08ea56a775c69c5f214fc6b99e9e6b1a238e6729504ee72e49046094b4a5fa10`.

The official validator accepts all 199,319 rows. A rebuild from raw tactical assets with a fresh cache and the distributed model reproduced the exact file. Five integrity tests cover label-independent corruption, team exchange, target exclusion, halftime boundaries and scorer/label mapping. Evidence is in `cook/reports/v2/clean_reproduction.json` and `cook/reports/release.json`.

## Reproduce training and evaluation

The frozen earlier prototype is included for comparison. Download all seven openly licensed external matches and rebuild the final experiment:

```bash
.venv-barcelonai/bin/python -m cook.get_external
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m cook.improve --stage train
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m cook.improve --stage test
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m cook.submit
.venv-barcelonai/bin/python -m unittest cook.test_integrity -v
```

Training also requires unlabelled competition tactical tracking to estimate corruption durations. Fetch URLs and local checksums are recorded by `cook.get_external`. The full feature list, transition matrix and selected configuration are in `cook/reports/v2/frozen_config.json`. Reports directly under `cook/reports/` describe the older prototype; final reports are under `cook/reports/v2/`. Do not use the legacy `cook.train --stage predict` command to generate the final submission.

The public repository excludes competition video, raw tracking, caches and the generated viewer. The organisers' framework and scorer retain their licence and attribution. External pretrained-model provenance is documented in `models/README.md`.
