# BarcelonAI

Two approaches to possession inference for More than a Hack 2027. Method 1 is the v9 tracking/state model. The final submission uses Method 2: an externally trained gradient-boosted classifier followed by sequence decoding. It uses 120 features and does not blend Method 1 predictions.

Read the [four-page report](output/pdf/BarcelonAI.pdf) or [text version](BarcelonAI.md) for a plain-language explanation, both methods, published baselines and fair comparisons. [Technical details](TECHNICAL_DOCUMENT.md) include the data split and retraining commands.

## Generate the final submission

Use Python 3.13. Put the organisers' licensed tactical ATD files in `data/tactical/` using the layout in `DATA.md`. Method 2 needs no video or broadcast tracking. Do not commit competition data.

```bash
python3.13 -m venv .venv-barcelonai
.venv-barcelonai/bin/python -m pip install -r requirements-cook.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m cook.submit
.venv-barcelonai/bin/python evaluation/validate_submission.py evaluation/results/BarcelonAI.csv
```

The selected model is included in `models/v2_leaf15.joblib`. A fresh-cache rebuild from raw tactical assets reproduced the exact 199,319-row file. Expected CSV SHA-256: `08ea56a775c69c5f214fc6b99e9e6b1a238e6729504ee72e49046094b4a5fa10`.

## Results

All values below are spellF1, the main ranking metric; they are not percentage accuracy.

| Evaluation | Official smooth50 | Method 1 | Method 2 |
|---|---:|---:|---:|
| External test J03WOY | 0.1812 | 0.2595 | 0.5963 |
| External test J03WR9 | 0.1381 | 0.1973 | 0.5917 |
| Competition public five-minute sanity check | 0.3256 | 0.4353 | 0.5622 |

Each row compares methods on the same input. External tests simulate tracking errors; Method 1 cannot borrow broadcast observations on those matches. The public sample did not select the model and has wide uncertainty. The official private full-match score is unknown. The report lists all four organisers' published full-match baselines separately, including the 0.2433 smooth50 floor.

## Local viewer

The private generated viewer is excluded from this repository. To compare both methods with smooth50, additionally supply the licensed broadcast ATD and MP4, then run:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m mtah.pipeline --name v9
.venv-barcelonai/bin/python -m cook.viewer
.venv-barcelonai/bin/python -m cook.serve
```

Open `http://127.0.0.1:8765/data/viewer/index.html`. Keep this generated viewer and the licensed match assets private. It links to the final CSV and report.

## Reproduction records

Final configuration, tests and fresh-cache evidence are in `cook/reports/v2/`; `cook/reports/release.json` identifies the submitted candidate. Older reports directly under `cook/reports/` describe the earlier learned prototype, not the final model. Use `cook.submit` for final inference; `cook.train` reproduces the older prototype.

The official starter framework is MIT-licensed. External data and pretrained model provenance are in [models/README.md](models/README.md). [SUBMISSION_READY.md](SUBMISSION_READY.md) lists the form details. No entry has been submitted by this code.
