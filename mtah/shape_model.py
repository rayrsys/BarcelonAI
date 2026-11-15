"""Self-supervised possession-from-shape model.

Pseudo-labels come from an algorithm, never a person: frames where the tracked ball sits within
1 m of a player and clearly closer to one team (margin > 1.5 m) while play is moving. The model is
trained on team geometry only -- no ball features -- so it can be applied when the ball is lost.
"""
import os, sys, json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.features import load as load_features, roll_mean, FPS  # noqa: E402
from mtah.atd import CACHE, ball_mode  # noqa: E402

SHAPE = ["cx", "sx", "sy", "line", "front", "vx", "in_opp_half"]


SHAPE2 = SHAPE + ["area", "depth", "press"]


def design(f, version=1):
    cols, names = [], []
    for s in (SHAPE if version == 1 else SHAPE2):
        h, a = f["h_" + s], f["a_" + s]
        cols += [h, a, h - a]; names += ["h_" + s, "a_" + s, "d_" + s]
    # smoothed versions of the most dynamic signals
    for s in ("vx", "cx"):
        for w in (FPS, 3 * FPS):
            cols.append(roll_mean(f["h_" + s] - f["a_" + s], w)); names.append(f"d_{s}_r{w}")
    cols.append(f["n_players"]); names.append("n_players")
    if version >= 2:
        cols.append(f["overlap"]); names.append("overlap")
        for s in ("press", "area"):
            cols.append(roll_mean(f["h_" + s] - f["a_" + s], 2 * FPS)); names.append(f"d_{s}_r50")
    return np.column_stack(cols).astype(np.float32), names


def pseudo_labels(f, version=1):
    dmin = np.fmin(f["d_home"], f["d_away"]); margin = f["d_away"] - f["d_home"]
    if version != 2:   # v1 and v3 use the strict labels; v2 tries more, noisier ones
        ok = (f["ball_present"] == 1) & (dmin < 1.0) & (np.abs(margin) > 1.5) & (f["speed_mean_1s"] > 1.3) & (f["half"] > 0)
    else:  # more, slightly noisier labels
        ok = (f["ball_present"] == 1) & (dmin < 1.5) & (np.abs(margin) > 1.0) & (f["speed_mean_1s"] > 1.2) & (f["half"] > 0)
    y = (margin > 0).astype(np.int8)  # 1 = home
    return ok, y


def fit(f, X=None):
    X = design(f)[0] if X is None else X
    ok, y = pseudo_labels(f)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=0)
    m.fit(X[ok], y[ok])
    return m


def blocked_cv(f, X, block_s=300, version=1):
    """Out-of-fold P(home) for every frame, 5-minute contiguous folds (no temporal leakage)."""
    ok, y = pseudo_labels(f, version)
    n = len(y); blk = np.arange(n) // (block_s * FPS); folds = blk % 5
    p = np.full(n, np.nan)
    for k in range(5):
        tr = ok & (folds != k)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        te = folds == k
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def load_oof(rebuild=False, version=1):
    suf = "_fill" if ball_mode() == "fill" else ""
    path = os.path.join(CACHE, (f"shape_oof{suf}.npy") if version == 1 else f"shape_oof_v{version}{suf}.npy")
    if rebuild or not os.path.isfile(path):
        f = load_features(); X, _ = design(f, version); np.save(path, blocked_cv(f, X, version=version))
    return np.load(path)


if __name__ == "__main__":
    import pandas as pd
    f = load_features(); X, names = design(f)
    ok, y = pseudo_labels(f)
    p = load_oof(rebuild=True)
    graded = f["half"] > 0
    out = {"n_pseudo": int(ok.sum()), "pseudo_home_share": round(float(y[ok].mean()), 3),
           "pseudo_share_of_graded": round(float(ok[graded].mean()), 3)}
    acc = lambda pr, t, m: round(float(((pr[m] > 0.5) == t[m]).mean()), 3)
    out["oof_acc_on_pseudo"] = acc(p, y, ok)
    ps = roll_mean(p, 2 * FPS)
    out["oof_acc_on_pseudo_smoothed2s"] = acc(ps, y, ok)
    conf = ok & (np.abs(p - 0.5) > 0.3)
    out["oof_acc_when_conf>0.8"] = acc(p, y, conf); out["share_conf>0.8"] = round(float(conf[ok].mean()), 3)
    gt = pd.read_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv")["label"].values
    alive = np.isin(gt, ["H", "A"]); t = gt == "H"
    out["public_check_acc"] = acc(p, t, alive)
    out["public_check_acc_smoothed2s"] = acc(ps, t, alive)
    out["public_check_acc_ball_missing"] = acc(ps, t, alive & (f["ball_present"] == 0))
    out["public_n_alive_ball_missing"] = int((alive & (f["ball_present"] == 0)).sum())
    # nearest-team baseline accuracy on the same public frames for reference
    near = f["d_home"] < f["d_away"]; mm = alive & (f["ball_present"] == 1)
    out["public_check_nearest_team_acc"] = round(float((near[mm] == t[mm]).mean()), 3)
    os.makedirs("eda/out", exist_ok=True); json.dump(out, open("eda/out/shape_model.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
