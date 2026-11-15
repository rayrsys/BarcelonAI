# BarcelonAI — Possession persists when the ball disappears

More than a Hack 2027 · Phase 1 · Technical submission document · 15 November 2026

**Team members:** Sayfullah Jumoorty; Hamdullah Dadabhoy; Muaaz Bassa; Rayhaan Hanslod; Muzzimmil Sakidas.

## Approach and motivation

The tactical ball track is missing on 40.7% of graded frames. Our independent, label-free EDA found 150 missing-ball intervals, a median duration of 10.72 seconds, and 99.7% of missing time in gaps longer than five seconds. This motivates learning football state across long occlusions, rather than simply smoothing nearest-player decisions. The intended output is a sequence of controlled possessions and stoppages.

BarcelonAI combines a supervised histogram gradient-boosted classifier trained on **other matches** with the existing v9 possession system. The new classifier uses 100 tracking-derived features: team geometry, movement, ball proximity, gap duration, short temporal context, and the most recent and next credible control observations. The task is offline, so future observations are available and used. No features identify a particular frame or player.

The classifier contributes 75% of the final state probability; v9 contributes 25%. A Viterbi decoder applies transition probabilities estimated from the external training matches. All these choices were fixed using a separate external development match, before the external test and competition public sanity check.

## Assets used

Tactical ATD supplies normalized player/ball positions and player speed. Ball position is converted to metres using the published pitch dimensions. Team features are invariant to track-slot identity; transient IDs are not treated as persistent player identities. The inherited v9 pipeline also uses broadcast ATD after its automatic ball/player geometry alignment verifies an offset. It borrows broadcast ball observations only for team evidence, retaining the tactical source for liveness inference.

The new classifier itself uses tactical features only. Homography and video are not used for model fitting or inference. Broadcast video appears only in the private local comparison viewer; unsynchronised sections show no prediction. Smart Tagging was explored in aggregate but excluded from the final classifier. Its agreement with algorithmically selected clear-control frames varied by tag, and matching tags are not available in the external training data. Tags are not treated as ground truth.

## Training data and provenance

We use the open IDSSE dataset: Bassek, Rein, Weber and Memmert, *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195 (2025). [Paper](https://doi.org/10.1038/s41597-025-04505-y), [dataset](https://doi.org/10.6084/m9.figshare.28196177), [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/).

These are German league matches, not the supplied Barcelona–Athletic match. Targets come from the dataset's provided `BallStatus` and `BallPossession` fields: inactive → D; active, home ownership → H; active, away ownership → A. Unknown ownership is ungraded. We did not annotate any frames. The data provider's tracking and quality-control process is described in the cited paper; these are provided labels, not labels newly created by our team.

| Role | Match IDs | Use |
|---|---|---|
| Training | J03WPY, J03WN1 | Classifier and transition counts |
| Development | J03WMX | Select tree size, mixture weight and evidence scaling |
| Untouched test | J03WOH | Evaluate the frozen system once |
| Competition match | BAR–ATH | Unlabelled inference and inherited algorithmic self-training only |

The two training matches contain 287,772 labelled frames in total. Training samples every fifth frame, uses two deterministic sensor corruptions, and augments each sample by exchanging home/away teams. This produces 230,224 training rows, not 230,224 independent observations. No Barcelona public or private ground truth enters training or selection. The inherited shape model learns only from algorithm-generated pseudo-labels on the unlabelled competition tracking.

Downloads use public Figshare file IDs, with URLs and SHA-256 checksums recorded locally. Coordinates are converted from centred metres to normalized pitch coordinates; player speed is converted from km/h to m/s. A separating ungraded frame prevents external possessions crossing halftime.

## Noise, control and stoppages

External tracking is cleaner than automatic competition tracking. We therefore alternate visible and hidden ball stretches using run lengths measured from unlabelled competition tracking, add temporally correlated spatial noise, and remove random player tracks in two-second blocks. Random seeds are fixed. Corruption never reads possession targets. This is a controlled stress test, not a perfect simulation of Metrica errors.

A credible ball-control observation requires a visible ball within 2 m of a player, a nearest-opponent distance margin greater than 1.5 m, ball speed below 12 m/s, and the ball more than 0.3 m inside the pitch boundary. The feature vector records the last/next controlling side, time to each observation, and whether they agree. These observations are features, not hard overrides. A loose ball or brief opposing touch need not change ownership. The sequence model can retain possession until sustained evidence supports a change.

Dead-ball inference combines learned movement/geometry patterns with the inherited v9 liveness, gap-mixture, kickoff and restart detectors. Missing ball alone is not defined as dead. D breaks possession; H–D–H represents two possessions. Each half is decoded separately, with X outside the published grading intervals.

The new model uses 220 boosting iterations, learning rate 0.06, 15 maximum leaves, minimum 60 samples per leaf and L2 regularization 8. It has no random early-stopping split. Predictions sampled at 5 Hz are interpolated to 25 Hz before final decoding. Transition counts are symmetrized between H/A. Full feature names and selected parameters are in `cook/reports/frozen_config.json`.

## Evaluation

All scores below use the organisers' unmodified metric library. The external development search comprised 24 configurations: 2 tree sizes × 3 mixture weights × 4 evidence scales. The selected configuration was saved before reading the test score. External competitors see the same corrupted tracking. The v9 adapter retains its geometry, EM, pseudo-label shape model and self-training; broadcast borrowing is disabled because those external matches have no broadcast asset. This compares algorithms on a common external input, not the exact input available in Barcelona.

| Untouched external test J03WOH | spellF1 | macroF1 | Predicted possessions |
|---|---:|---:|---:|
| Official smooth50 algorithm | 0.1735 | 0.4181 | 490 |
| Previous model v9, external adapter | 0.2470 | 0.6252 | 189 |
| **BarcelonAI / BarcelonAI** | **0.5279** | **0.8864** | **264** |

The external test has 402 true possessions. The new model improves spellF1 by 0.2809 absolute; it still misses short spells. The scorer's 300-draw within-match bootstrap intervals are 0.485–0.572 for the new model and 0.213–0.287 for v9. These intervals do not quantify generalisation to new matches or providers.

After freezing the model and writing the submission, a public-window alignment/format check gave:

| Public five-minute sanity check | spellF1 | macroF1 |
|---|---:|---:|
| Official smooth50 | 0.3256 | 0.6207 |
| Previous model v9 saved submission | 0.4353 | 0.6913 |
| **BarcelonAI** | **0.5529** | **0.7969** |

This is **not a leaderboard or tuning result**. Its 1,000-draw spellF1 intervals overlap: 0.388–0.718 versus 0.294–0.576. No model was changed or selected because of these public scores. The published full-match baseline floor, 0.2433, is measured on a different, private evaluation set and must not be directly compared with our public or external scores. Our private full-match result is unknown.

## Reproduction and verification

Python 3.13, with exact package versions in `requirements-cook.txt`. The source package includes the selected externally trained model under `models/`; it contains no competition video or tracking. With the supplied tactical/broadcast ATD assets placed in the documented `data/` layout:

```bash
python3.13 -m venv .venv-barcelonai
.venv-barcelonai/bin/python -m pip install -r requirements-cook.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m cook.train --stage predict
.venv-barcelonai/bin/python evaluation/validate_submission.py evaluation/results/BarcelonAI.csv
```

Expected CSV SHA-256: `05a2f57cbc7faa660ae0660c80e6b5d7e12b12af3aede25c2ff31d53aec33432`.

The file contains 199,319 rows and only H/A/D/X. Repeated inference produced identical CSV bytes. A separate fresh-cache run rebuilt ATD parsing, features, broadcast fill and shape pseudo-models from raw competition assets and reproduced all labels exactly (zero differing frames). Five integrity tests passed, covering label-independent corruption, team-swap consistency, target exclusion from features, halftime boundaries and metric/label mapping. The regenerated v9 labels also matched the saved v9 file exactly.

To retrain and independently reproduce the experimental process:

```bash
.venv-barcelonai/bin/python -m cook.get_external J03WPY J03WN1 J03WMX J03WOH
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m cook.train
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m cook.train --stage test
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m cook.train --stage predict
.venv-barcelonai/bin/python -m unittest cook.test_integrity -v
```

## Limitations and next work

Only two matches train the new classifier, one selects parameters and one tests it. Different matches may include the same clubs; this is match separation, not guaranteed club separation. Tracking providers and definitions of control can differ. Synthetic missingness is label-independent, whereas real occlusions correlate with play and stoppages. Random player loss does not reproduce every identity switch or camera error. The model uses future observations and is not suitable for a strictly live stream without modification. Short possessions and ambiguous deflections remain difficult. More independent labelled matches and more realistic sensor corruption are the most useful next steps.

The prior v9 component is retained as part of the ensemble, rather than presenting the full pipeline as a new implementation. The organisers’ starter framework and metric library retain their original attribution and licence. All final labels are algorithmic. No hand corrections or frame-specific exceptions were added.
