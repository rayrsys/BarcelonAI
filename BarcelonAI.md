# BarcelonAI

## What we are trying to do

In football, two teams try to move a ball towards the other team's goal. A team has **possession** when it controls the ball. Control can continue while the ball travels between teammates. When play stops, for example because the ball goes out of bounds, neither team has a live possession.

The challenge is to identify who has possession, or whether play has stopped, at every moment of one match. Our output contains 199,319 labels: one for each video frame, recorded 25 times per second. H means the home team; A means the away team; D means stopped play; X marks periods outside our labelled halves.

The organisers provide automatically measured player and ball positions. The ball is missing from **40.7%** of the frames that count towards the score. A missing measurement does not mean the game stopped. The camera or tracking system may simply have lost sight of the ball.

Our method uses player movement, available ball observations, and evidence before and after a gap to estimate the state of play.

## How well does it work?

The main score is **spellF1**. A spell is one continuous period in which a team controls the ball. The score rewards recovering these periods, including their timing, without inventing extra possessions. Higher is better; 1.0 is perfect. It is not the percentage of individual frames labelled correctly.

The organisers' baseline mostly chooses the team with the player closest to the ball, then smooths its decisions over time. We compare it with our previous v9 system and the new BarcelonAI model.

| Method | Public five-minute check | Separate external test |
|---|---:|---:|
| Organisers' baseline (smooth50) | 0.3256 | 0.1735 |
| Previous model v9 | 0.4353 | 0.2470 |
| **BarcelonAI** | **0.5529** | **0.5279** |

Compare methods **within each column**. The columns describe different tests. On the external test, the new model also improved the score for individual frame labels, macroF1, from 0.6252 to 0.8864 compared with v9.

The public sample was checked only after fixing the model; it did not choose our settings. It contains just 17 possessions, and the old and new models' uncertainty intervals overlap. The external match was kept aside until the model was fixed. All methods received the same simulated tracking loss; v9 could not use its optional second camera because none was supplied for that match.

The organisers' published **private full-match baseline is 0.2433**. Our score on that private test is unavailable. We do not claim to know the official competition ranking.

## Why the new method helps

**Learn from other matches.** We use two public German league matches with existing possession and play-status labels. The model learns relationships between ball proximity, player movement and the state of play. No human labels from the competition match train this classifier.

**Practise with missing information.** We hide ball observations for varying lengths of time, remove some player observations and add small position errors. Gap lengths come from the unlabelled competition tracking. The corruption process never reads possession labels. This matters because 99.7% of missing-ball time in the competition data lies in gaps longer than five seconds.

**Use evidence across time.** If the same team controls the ball before and after a gap, that supports continued possession. It is evidence, not a guarantee: player movement and other observations also matter. Since the full recording is available, the method can look backwards and forwards.

**Separate a hidden ball from stopped play.** Players often move differently when the game pauses. Those patterns help the model avoid labelling every missing-ball interval as a stoppage.

**Make a consistent sequence.** We combine 75% of the new model's probabilities with 25% from the previous v9 system. A final decoding step discourages rapid, unsupported switches. A brief touch by an opponent need not become a new possession.

## How we tested it

| Stage | Match IDs | Purpose |
|---|---|---|
| Training | J03WPY, J03WN1 | Learn the model and state-change rates |
| Development | J03WMX | Choose from 24 configurations |
| Untouched test | J03WOH | Evaluate the fixed model |
| Competition sample | Public five minutes | Check the completed pipeline only |

We used the organisers' unchanged scoring code. The external test contains 402 true possessions; BarcelonAI predicted 264. The improved score therefore does not mean the task is solved. Short or ambiguous possessions remain difficult.

The external test's within-match spellF1 bootstrap intervals are 0.485–0.572 for BarcelonAI and 0.213–0.287 for v9. They do not measure uncertainty across different matches or data providers. Only one external match was reserved for final testing. Simulated errors cannot reproduce every real camera failure, and possession definitions may vary between providers. The model also uses future observations, so it is intended for completed matches, not strictly live use.

## Technical details and data sources

The new component is a **histogram gradient-boosted classifier**: a learned collection of small decision trees. It uses 100 features covering ball proximity, previous and next credible control, player movement, team shape and short time windows. The selected model has 220 iterations, learning rate 0.06, at most 15 leaves per tree, minimum 60 samples per leaf and regularization 8.

The training matches contain 287,772 labelled frames. Sampling every fifth frame, two fixed sensor corruptions and home/away exchange augmentation produce 230,224 training rows; these are not independent observations. The v9 component uses a Hidden Markov Model, which combines evidence with the tendency for a state to persist, plus algorithm-generated shape labels and automatic stoppage/restart detectors. A final Viterbi decoder finds a consistent H/A/D sequence. Each half is processed separately. All output is generated by code, without hand corrections.

The new classifier uses tactical tracking. The v9 component can borrow broadcast ball observations after automatic alignment checks. Video is only used in the private viewer. Homography is not used. Automatic tactical tags were explored but excluded because reliability varied and the external matches have no equivalent tags.

External data: Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*. The dataset supplies BallPossession and BallStatus labels; our team did not annotate those matches. Our changes comprise parsing, unit conversion, simulated errors, feature extraction and training. [Paper](https://doi.org/10.1038/s41597-025-04505-y), [dataset](https://doi.org/10.6084/m9.figshare.28196177), [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/).

## Reproducing the submission

Code: [github.com/rayrsys/BarcelonAI](https://github.com/rayrsys/BarcelonAI). The repository includes the selected model, exact package versions and full technical notes. Supply the private competition tracking separately in the documented data folder. Video is not needed for inference.

```bash
python3.13 -m venv .venv-barcelonai
.venv-barcelonai/bin/python -m pip install -r requirements-cook.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv-barcelonai/bin/python -m cook.train --stage predict
.venv-barcelonai/bin/python evaluation/validate_submission.py \
  evaluation/results/BarcelonAI.csv
```

The CSV passes the official 199,319-row validator. Repeated runs produce identical bytes. A fresh rebuild of all inherited caches from raw assets reproduced every prediction. Five integrity checks cover label-independent corruption, team exchange, feature/target separation, halftime boundaries and score mapping.

Expected CSV SHA-256: `05a2f57cbc7faa660ae0660c80e6b5d7e12b12af3aede25c2ff31d53aec33432`.

**Team:** Sayfullah Jumoorty, Hamdullah Dadabhoy, Muaaz Bassa, Rayhaan Hanslod and Muzzimmil Sakidas.

Prepared for More than a Hack 2027, Phase 1. Implementation details and retraining commands are in TECHNICAL_DOCUMENT.md.
