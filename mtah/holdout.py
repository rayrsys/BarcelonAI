"""Held-out touch test: a label-free check of possession inference.

Clear ball-control moments (ball within 0.7 m of a player, >2 m closer to him than to any opponent,
ball slow) are grouped into touch clusters. A random 20% of clusters have ALL ball evidence removed
within +/-0.5 s; the model is decoded; we score whether the label at each hidden cluster names the
controlling team. 'turnover' clusters are those whose team differs from the previous cluster's --
the possession changes that spellF1 rewards. Shape evidence is out-of-fold, so nothing leaks.
"""
import numpy as np
from mtah.features import FPS


def clusters(f):
    dmin = np.fmin(f["d_home"], f["d_away"]); margin = f["d_away"] - f["d_home"]
    ctrl = (f["ball_present"] == 1) & (dmin < 0.7) & (np.abs(margin) > 2.0) & (np.nan_to_num(f["ball_speed"], nan=99) < 4) & (f["half"] > 0)
    # a ball held outside the field (thrower, ball boy) is out of play by definition: not a possession touch
    ctrl &= np.nan_to_num(f["ball_edge_m"], nan=-1) > 0.3
    idx = np.where(ctrl)[0]
    if not len(idx):
        return []
    br = np.where(np.diff(idx) > 5)[0]
    starts = np.concatenate(([idx[0]], idx[br + 1])); ends = np.concatenate((idx[br], [idx[-1]])) + 1
    out = []
    for a, b in zip(starts, ends):
        team = "H" if np.mean(margin[a:b] > 0) >= 0.5 else "A"
        out.append((a, b, team))
    return out


def masks(f, seed, frac=0.2):
    cl = clusters(f); rng = np.random.default_rng(seed)
    pick = rng.random(len(cl)) < frac
    hide = np.zeros(len(f["half"]), bool); held = []
    prev_team = None; prev_half = None
    for i, ((a, b, t), p) in enumerate(zip(cl, pick)):
        h = f["half"][a]
        turnover = prev_team is not None and prev_half == h and t != prev_team
        nxt = cl[i + 1] if i + 1 < len(cl) and f["half"][cl[i + 1][0]] == h else None
        # sustained: the new team has the next clear touch too; isolated: it goes straight back
        kind = None if not turnover or nxt is None else ("sust" if nxt[2] == t else "isol")
        if p:
            hide[max(a - FPS // 2, 0):b + FPS // 2] = True; held.append((a, b, t, turnover, kind))
        prev_team, prev_half = t, h
    return hide, held


def score(labels, held):
    hit = [np.mean(labels[a:b] == t) for a, b, t, _, _ in held]
    tov = [np.mean(labels[a:b] == t) for a, b, t, tv, _ in held if tv]
    sust = [np.mean(labels[a:b] == t) for a, b, t, _, k in held if k == "sust"]
    isol = [np.mean(labels[a:b] == t) for a, b, t, _, k in held if k == "isol"]
    m = lambda x: float(np.mean(x)) if x else np.nan
    return {"ho_acc": m(hit), "ho_turnover_acc": m(tov), "ho_sust_acc": m(sust), "ho_isol_flip": m(isol),
            "ho_n": len(held), "ho_n_turnover": len(tov), "ho_n_sust": len(sust), "ho_n_isol": len(isol)}


def nll(post, held):
    """Mean negative log posterior of the true team at hidden touches (lower is better), and Brier."""
    ll, br = [], []
    for a, b, t, _, _ in held:
        p = np.nanmean(post[a:b, 0 if t == "H" else 1])
        ll.append(-np.log(max(p, 1e-4))); br.append((1 - p) ** 2)
    return {"ho_nll": float(np.mean(ll)), "ho_brier": float(np.mean(br))}
