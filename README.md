# BarcelonAI

Possession inference for More than a Hack 2027. The model combines externally supervised gradient boosting (75%) and the prior v9 state model (25%), then decodes a three-state sequence.

## Generate the submission

Use Python 3.13. Put the organisers' licensed ATD files in `data/tactical/` and `data/broadcast/` using the file layout in `DATA.md`. Videos are unnecessary for inference. Do not commit competition data.

```bash
python3.13 -m venv .venv-barcelonai
.venv-barcelonai/bin/python -m pip install -r requirements-cook.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m cook.train --stage predict
.venv-barcelonai/bin/python evaluation/validate_submission.py evaluation/results/BarcelonAI.csv
```

The externally trained model is included in `models/`. On a fresh checkout, feature and pseudo-model caches are generated automatically from the supplied match assets. Expected CSV SHA-256: `05a2f57cbc7faa660ae0660c80e6b5d7e12b12af3aede25c2ff31d53aec33432`.

Read the [plain-language PDF](output/pdf/BarcelonAI.pdf) or [text version](BarcelonAI.md) for the method, external data attribution, results, limitations and full retraining commands. [SUBMISSION_READY.md](SUBMISSION_READY.md) records the outstanding team/repository/form requirements.

## Results and their limits

| Evaluation | Official smooth50 | Previous v9 | BarcelonAI |
|---|---:|---:|---:|
| Untouched external match, same simulated sensor loss | 0.1735 | 0.2470 | 0.5279 |
| Competition public five-minute sanity check | 0.3256 | 0.4353 | 0.5529 |

The public window was not used for tuning; its confidence intervals overlap. The official full-match score is unknown. The external comparison uses v9 without broadcast borrowing because the external dataset has no broadcast asset. The new system remains imperfect and undercounts possessions in the external test.

## Local viewer

The generated video viewer is private and excluded from the source package. To regenerate it after inference, generate the v9 file and the supplied baselines, then build the viewer:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-barcelonai/bin/python -m mtah.pipeline --name v9
.venv-barcelonai/bin/python -m cook.viewer
.venv-barcelonai/bin/python -m cook.serve
```

Open `http://127.0.0.1:8765/data/viewer/index.html`. This requires the local broadcast MP4. No match video is uploaded. The viewer compares new/old/baseline labels and links to the final CSV and technical document.

The official starter framework is MIT-licensed. External data and pretrained model provenance are detailed in `models/README.md`.
