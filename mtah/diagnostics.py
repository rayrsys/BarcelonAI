"""Label-free diagnostics for a label array (no ground truth of any kind)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from starter.load_smart_tagging import load as load_tags  # noqa: E402
from mtah.features import load as load_features  # noqa: E402

_cache = {}


def _ctx():
    if not _cache:
        f = load_features(raw=True)
        tags = load_tags()
        _cache["sp"] = np.clip(np.array([t["start_frame"] for t in tags if t["code"] == "SET PIECES"]), 0, len(f["half"]) - 1)
        _cache["graded"] = f["half"] > 0
        dmin = np.fmin(f["d_home"], f["d_away"]); margin = f["d_away"] - f["d_home"]
        ctrl = (f["ball_present"] == 1) & (dmin < 0.7) & (np.abs(margin) > 2.0) & (np.nan_to_num(f["ball_speed"], nan=99) < 4) & _cache["graded"]
        _cache["ctrl"] = ctrl; _cache["ctrl_team"] = np.where(margin > 0, "H", "A")
        from mtah.holdout import clusters
        _cache["clusters"] = clusters(f)
    return _cache


def diagnose(labels):
    c = _ctx(); g = labels.astype(str).copy(); g[~c["graded"]] = "X"
    isd = g == "D"; near = isd.copy()
    for k in range(1, 126):
        near[k:] |= isd[:-k]
    gg = g[c["graded"]]; ch = np.concatenate(([True], gg[1:] != gg[:-1])); segs = gg[ch]
    lens = np.diff(np.concatenate((np.where(ch)[0], [len(gg)])))
    poss = np.isin(segs, ["H", "A"])
    alive = np.isin(g, ["H", "A"]) & c["ctrl"]
    # possessions that contain at least one clear touch by the team they credit (precision proxy)
    idx_g = np.where(c["graded"])[0]; starts = idx_g[np.where(ch)[0]]; ends = np.concatenate((starts[1:], [idx_g[-1] + 1]))
    touch = np.zeros((2, len(g)), np.int32)
    for a, b, t in c["clusters"]:
        touch[0 if t == "H" else 1, a] += 1
    cum = np.concatenate((np.zeros((2, 1), np.int64), np.cumsum(touch, axis=1)), axis=1)
    sup = []
    for s_, e_, lab in zip(starts, ends, segs):
        if lab in ("H", "A"):
            k = 0 if lab == "H" else 1; sup.append(cum[k, e_] - cum[k, s_] > 0)
    return {
        "possessions": int(poss.sum()), "mean_poss_s": round(float(lens[poss].mean() / 25), 1),
        "poss_lt2s": round(float(np.mean(lens[poss] < 50)), 3),
        "dead_spells": int((segs == "D").sum()), "dead_share": round(float(np.mean(gg == "D")), 3),
        "restart_recall": round(float(near[c["sp"]].mean()), 3),
        "control_agree": round(float(np.mean(g[alive] == c["ctrl_team"][alive])), 3),
        "control_dead": round(float(np.mean(g[c["ctrl"]] == "D")), 3),
        "supported": round(float(np.mean(sup)), 3), "unsupported_n": int(len(sup) - np.sum(sup)),
    }
