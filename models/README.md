# BarcelonAI pretrained models

## Final submission: Method 2

`v2_leaf15.joblib` is the selected standalone histogram gradient-boosted classifier. It trains on IDSSE matches J03WPY, J03WN1, J03WMX and J03WOH, with deterministic synthetic tracking corruption and home/away exchange augmentation. J03WQQ selects the configuration; J03WOY and J03WR9 are held-out tests. See `cook/reports/v2/frozen_config.json` for features and parameters.

Model SHA-256: `5268d8d413db5cf6faa056171847056c7f719a75296b95a0a412cb425cb986a0`.

Run `python -m cook.submit` for the final submission. It uses this model and learned transition probabilities; it does not blend Method 1 predictions.

## Earlier prototype

`cook_boost_15.joblib` is retained to reproduce the frozen earlier prototype used as a comparator during development. It trains on J03WPY and J03WN1 only. It is not the final submission model.

Model SHA-256: `84ff289b0c097b8b5db13214b4581b68ea41646162d08932daed4281aabbac9e`.

## Provenance

Neither model fits human labels from the competition match. Ball-gap duration statistics come from unlabelled competition tracking; classifier training examples come from the external matches.

Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025), [An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y). [Dataset](https://doi.org/10.6084/m9.figshare.28196177), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Our changes comprise parsing, coordinate/unit normalization, simulated tracking corruption, feature extraction, augmentation and model fitting. No raw data is included here.

Load only trusted joblib files: deserialization can execute code. Final inference verifies the model SHA-256 before loading. Dependencies are pinned in `requirements-cook.txt`; retraining is documented in `TECHNICAL_DOCUMENT.md`.
