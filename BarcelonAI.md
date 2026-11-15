# BarcelonAI

## What the project does

In football, two teams try to move a ball towards the other team's goal. A team has **possession** when it controls the ball. Control can continue while the ball travels between teammates. When the referee stops play, neither team has a live possession.

This challenge asks us to identify which team controls the ball, or whether play has stopped, at every moment of a match. We produce 199,319 labels: one for each video frame, recorded 25 times per second. H means the home team, A the away team, D stopped play, and X periods outside the halves we label.

The organisers provide automatically measured player and ball positions. They also provide a simple reference method, called a **baseline**, so teams can measure whether their approach is an improvement. The baseline mostly picks the team with the player closest to the ball. Smoothed versions combine nearby decisions to reduce rapid switching.

The difficulty is that the ball is missing from **40.7%** of scored frames. Our analysis found that 99.7% of this missing time is in gaps longer than five seconds. A missing ball measurement does not mean the game stopped: the tracking system may simply have lost sight of it.

## Our two methods

**Method 1: infer the state from tracking and movement.** The v9 system combines ball-to-player distance, team shape, player movement and automatic restart detectors. It learns some patterns from confident, algorithm-generated examples in the unlabelled match. A state model then favours consistent periods of control over unsupported moment-to-moment switches. It can borrow ball observations from the second camera when automatic alignment verifies them.

**Method 2: learn from labelled examples in other matches.** The new model learns how tracking patterns relate to possession and stopped play from four public German league matches. We deliberately hide observations during training so it practises dealing with imperfect tracking. It considers player movement and credible control before and after a gap, then chooses a consistent sequence of states.

The methods share tracking-feature code, but use different learning approaches. **The final Method 2 does not blend in Method 1's predictions.** An earlier prototype did; the final configuration was chosen on a separate external development match. Method 2 is the recommended submission.

## How to read the scores

**spellF1** is the main ranking score. A spell is one continuous period of team control. The score rewards finding these periods with the right timing, without inventing too many extra possessions. **macroF1** measures individual frame labels while giving the three states equal weight. Both range up to 1.0; higher is better. Neither should be read as a simple percentage accuracy.

## Benchmarks and our results

### 1. The organisers' published full-match benchmarks

These are reproduced as published in EVALUATION.md. They use the organisers' private full-match answer key. **Our scores on that answer key are not available.**

| Published baseline | spellF1 | macroF1 |
|---|---:|---:|
| Raw | 0.0486 | 0.6532 |
| smooth25 | 0.1996 | 0.6752 |
| smooth50 - official floor | 0.2433 | 0.6724 |
| smooth125 | 0.2242 | 0.6356 |

### 2. A like-for-like check on the public five minutes

Every row below uses the same public sample and the unchanged official scorer. We checked our completed pipelines only after fixing their settings. These public scores did not select our models.

| Method | spellF1 | macroF1 |
|---|---:|---:|
| Official: raw | 0.0378 | 0.5972 |
| Official: smooth25 | 0.2868 | 0.6200 |
| Official: smooth50 | 0.3256 | 0.6207 |
| Official: smooth125 | 0.4242 | 0.5812 |
| Method 1: state model | 0.4353 | 0.6913 |
| Method 2: learned model | 0.5622 | 0.7989 |

Both methods score above the supplied baselines on this sample. Method 1's spellF1 gain over the strongest public baseline is small: 0.0111. Method 2 improves on Method 1 by 0.1269 and on smooth50 by 0.2366. These are differences in score, not percentage-point accuracy gains.

There are only 17 true possessions in the public sample. Uncertainty is wide and the models' intervals overlap. The public sample cannot establish the private ranking. Do not compare our public scores directly with the private scores in the first table.

## Testing on different matches

To obtain stronger evidence, we set aside two other matches before evaluating the final learned model. We removed ball observations in stretches, dropped some player tracks and added small position errors. This makes the test harder, although it cannot reproduce every real tracking failure. Every method received the same degraded tracking. Method 1's optional broadcast borrowing was disabled because these matches have no second-camera asset.

| Method | Test 1 spellF1 | Test 2 spellF1 | Mean macroF1 |
|---|---:|---:|---:|
| Official: smooth50 | 0.1812 | 0.1381 | 0.4371 |
| Method 1 | 0.2595 | 0.1973 | 0.5910 |
| Method 2 | 0.5963 | 0.5917 | 0.9019 |

Test 1 is J03WOY; Test 2 is J03WR9. The mean is an equal-weight average across matches. Method 2's mean spellF1 is **0.5940**, compared with **0.2284** for Method 1. It also improves on our earlier learned prototype on both matches: 0.5582 to 0.5963 and 0.5372 to 0.5917. The prototype's mean macroF1 was 0.8882; the final learned model reaches 0.9019.

### What improved, and what remains difficult?

Our earlier prototype handled long possessions better than short ones. We investigated finer time context, more training matches, and extra weighting for short possessions. The development test selected a model with more data and finer context; the extra weighting was not selected. This distinction matters: we report what the tests supported, not every idea we tried.

The final model still misses possessions. It predicts 353 where Test 1 has 456, and 304 where Test 2 has 370. Short control changes, deflections and ambiguous stoppages remain difficult. Improvement does not mean the task is solved.

### How we kept the comparison fair

Method 2 trained on J03WPY, J03WN1, J03WMX and J03WOH. J03WQQ selected one of 36 configurations. J03WOY and J03WR9 were untouched until the configuration was fixed. A predeclared rule promoted the candidate only if it beat the frozen earlier prototype on both tests. No weights were refitted using those test results. One public sanity check followed selection and did not change the recommendation.

The earlier prototype's original external test later became part of Method 2 training. We therefore do not reuse that earlier score as independent evidence about the final model. The report's final comparison uses only the two new tests.

## Technical details and reproduction

Method 1 uses a three-state Hidden Markov Model, which combines evidence with the tendency for a state to persist. It also uses algorithm-generated shape labels and automatic gap/restart detectors. Method 2 uses a histogram gradient-boosted classifier, a learned collection of small decision trees, followed by Viterbi sequence decoding.

The selected classifier has 120 features, 300 iterations, learning rate 0.06, at most 15 leaves per tree, minimum 60 samples per leaf and regularization 8. Features describe ball proximity, previous and next credible control, player movement, team shape and short time windows. Two fixed corruptions and home/away exchange augmentation produce 456,772 training rows; these are correlated examples, not independent observations. State-change probabilities are learned from the external training labels. Each half is decoded separately.

The final learned model uses **tactical tracking only**. It does not need broadcast video, broadcast tracking, homography or tactical tags. Method 1's optional broadcast tracking and the private video viewer remain available for comparison. Tactical tags were explored but excluded from the final model. All labels are generated by code; there are no manual corrections or rules tied to particular match frames.

External data: Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*. BallPossession and BallStatus are existing provider labels; our team did not annotate these matches. We parse and normalize the data, simulate tracking errors, extract features and fit the model. [Paper](https://doi.org/10.1038/s41597-025-04505-y), [dataset](https://doi.org/10.6084/m9.figshare.28196177), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Code and selected model: [github.com/rayrsys/BarcelonAI](https://github.com/rayrsys/BarcelonAI). Put the licensed tactical ATD assets in the documented data folder, then run:

```bash
python3.13 -m venv .venv-barcelonai
.venv-barcelonai/bin/python -m pip install -r requirements-cook.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1   .venv-barcelonai/bin/python -m cook.submit
.venv-barcelonai/bin/python evaluation/validate_submission.py   evaluation/results/BarcelonAI.csv
```

The 199,319-row CSV passes the official validator. Regenerating it from raw inputs, a fresh cache and the distributed pretrained model reproduces the exact file. Integrity tests cover corruption independence, team exchange, target/feature separation, halftime boundaries and score mapping.

Expected CSV SHA-256: `08ea56a775c69c5f214fc6b99e9e6b1a238e6729504ee72e49046094b4a5fa10`.

**Limits:** only two external matches provide the final test; clubs may overlap across different matches. Real camera errors and providers' definitions of control may differ. Future observations are used, so the method is for completed matches, not strictly live prediction. The private competition score remains unknown.

**Team:** Sayfullah Jumoorty, Hamdullah Dadabhoy, Muaaz Bassa, Rayhaan Hanslod and Muzzimmil Sakidas.

Prepared for More than a Hack 2027, Phase 1. Further implementation detail is in TECHNICAL_DOCUMENT.md.
