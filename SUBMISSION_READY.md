# BarcelonAI submission readiness

**Technical candidate ready. The public repository is published; final form upload remains.**

## Prepared

- `evaluation/results/BarcelonAI.csv` — the team-named submission, 199,319 rows; official format validator passed.
- `output/pdf/BarcelonAI.pdf` — four-page report, with `BarcelonAI.md` as its text version.
- `deliverables/BarcelonAI_source.zip` — allowlisted source package and externally trained weights for your public repository; no competition video, raw tracking, caches or generated viewer.
- `data/viewer/index.html` — private local results dashboard and three-model video comparison. Do not publish this generated file.
- `cook/reports/` — aggregate EDA, frozen experiment configuration, test metrics and reproducibility evidence.

The submission SHA-256 is `08ea56a775c69c5f214fc6b99e9e6b1a238e6729504ee72e49046094b4a5fa10`. Both repeat inference and a raw-asset rebuild reproduced the predictions exactly. Method 2 beat Method 1 on both held-out external matches (spellF1 0.5963 vs 0.2595 and 0.5917 vs 0.1973). The final public sanity check was 0.5622 vs 0.4353; public labels did not select the model. The private competition score is unknown.

## Required before upload

1. Five members supplied: **Sayfullah Jumoorty; Hamdullah Dadabhoy; Muaaz Bassa; Rayhaan Hanslod; Muzzimmil Sakidas**. Confirm these names and **BarcelonAI** match the registered entry.
2. Use the team repository: **https://github.com/rayrsys/BarcelonAI**. It includes the model, source code and reproduction instructions.
3. Review `BarcelonAI.md`, including the external-data attribution, method and limitations.
4. Use the submission form emailed to registered participants: team name **BarcelonAI**, possession file **BarcelonAI.csv**, public repository URL, and technical document **BarcelonAI.pdf**.
5. Submit by **17 November 2026**. The local rules do not state a time-of-day/timezone, so use the emailed deadline rather than assuming midnight. One response per team; the last edited response before the deadline is scored.

No competition entry has been uploaded. The authorised public code repository is https://github.com/rayrsys/BarcelonAI. Match-data redistribution is prohibited; publish the allowlisted source package, not the whole working folder.

## Open the local viewer

```bash
.venv-barcelonai/bin/python -m cook.serve
```

Open `http://127.0.0.1:8765/data/viewer/index.html`. It compares BarcelonAI, Method 1 and the official baseline. “Next disagreement” jumps to automatically found disagreements lasting at least one second. The broadcast has verified alignment on 68.4% of frames. “No sync” does not mean the competition submission is missing rows; the submission uses the full tactical timeline.

## Source rules

[Official submission instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md) · [Official rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md).

## Paste into the form

**Team name:** BarcelonAI

**Team members names:** Sayfullah Jumoorty; Hamdullah Dadabhoy; Muaaz Bassa; Rayhaan Hanslod; Muzzimmil Sakidas

**Possession file:** BarcelonAI.csv

**Technical document:** output/pdf/BarcelonAI.pdf (plain-language, four pages); BarcelonAI.md is the text version.

**Public repository URL:** https://github.com/rayrsys/BarcelonAI
