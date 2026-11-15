# BarcelonAI pretrained model

`cook_boost_15.joblib` is the frozen externally supervised component. It is trained on IDSSE matches J03WPY and J03WN1 only, with deterministic synthetic sensor corruption and team-swap augmentation. No human labels from the competition match enter its fitting. The empirical corruption-duration distribution is derived from unlabelled competition tracking; raw competition observations are not model features at training time.

Model SHA-256: `84ff289b0c097b8b5db13214b4581b68ea41646162d08932daed4281aabbac9e`.

Attribution: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025), [An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y). [Dataset DOI](https://doi.org/10.6084/m9.figshare.28196177). Data licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Our changes comprise parsing, coordinate/unit normalization, simulated sensor corruption, feature extraction, augmentation and model fitting. No raw data is included here.

Load only the trusted model supplied with this repository: joblib/pickle deserialization executes code. The inference command verifies its SHA-256 before loading it. Reproduction dependencies are in `requirements-cook.txt`; retraining is documented in `TECHNICAL_DOCUMENT.md`.
