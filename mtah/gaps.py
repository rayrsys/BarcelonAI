"""Gap-level dead-ball model: every ball-track gap is one event, classified alive/dead by an
unsupervised two-component Gaussian mixture on event features. No labels of any kind."""
import numpy as np
from sklearn.mixture import GaussianMixture


def gap_events(f):
    g = f["half"] > 0; miss = (f["ball_present"] == 0) & g
    m = np.concatenate(([0], miss.astype(np.int8), [0])); d = np.diff(m)
    return np.where(d == 1)[0], np.where(d == -1)[0]


def gap_features(f, S, E):
    n = len(f["half"])
    pre = np.clip(S - 1, 0, n - 1); post = np.clip(E, 0, n - 1)
    dur = (E - S) / 25
    tempo = np.array([np.nanmean(f["speed_mean"][a:b]) for a, b in zip(S, E)])
    spd_pre = np.array([np.nanmean(f["ball_speed"][max(a - 12, 0):a]) for a in S])
    X = np.column_stack([np.log(dur + 0.04), np.log(np.nan_to_num(tempo, nan=1.5) + 0.05),
                         np.log1p(np.clip(np.nan_to_num(f["ball_edge_m"][pre], nan=10), 0, None)),
                         np.log1p(np.clip(np.nan_to_num(f["ball_edge_m"][post], nan=10), 0, None)),
                         np.log1p(np.nan_to_num(spd_pre, nan=3))])
    return X


def dead_prob(f, seed=0):
    """(per-frame dead probability inside gaps (NaN elsewhere), per-gap table)."""
    S, E = gap_events(f); X = gap_features(f, S, E)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    gm = GaussianMixture(2, covariance_type="full", n_init=10, random_state=seed).fit(Z)
    dead_comp = int(np.argmax(gm.means_[:, 0]))          # the long-gap component is "dead"
    p = gm.predict_proba(Z)[:, dead_comp]
    pf = np.full(len(f["half"]), np.nan)
    for a, b, pi in zip(S, E, p):
        pf[a:b] = pi
    return pf, {"S": S, "E": E, "p": p, "X": X}
