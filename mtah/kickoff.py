"""Kickoff-formation detector, from tracking alone.

Before any kickoff (start of a half, after a goal) every player stands in his own half and the
ball sits on the centre spot. Frames in that formation are dead; the kickoff is the moment the
ball leaves the spot. Purely geometric -- no frame numbers, no labels.
"""
import numpy as np
from mtah.atd import load as load_atd, PITCH
from mtah.features import roll_mean, FPS


def formation(f, own_tol_m=1.0, min_players=16, min_share=0.95):
    d = load_atd(); P, B, team = d["players"], d["ball"], d["team"]
    X = P[:, :, 0] * PITCH[0]
    hd = f["home_dir"].astype(float)[:, None]
    # metres from own goal for every player, oriented per team
    own = np.where(team[None] == 0, np.where(hd > 0, X, PITCH[0] - X), np.where(hd > 0, PITCH[0] - X, X))
    tracked = np.isfinite(own)
    in_own = (own <= PITCH[0] / 2 + own_tol_m) & tracked
    n = tracked.sum(1); share = np.where(n > 0, in_own.sum(1) / np.maximum(n, 1), 0)
    bm = B[:, :2] * PITCH; centre = PITCH / 2
    bdist = np.linalg.norm(bm - centre, axis=1)
    ball_ok = ~np.isfinite(bdist) | (bdist < 3.0)       # at the spot, or not tracked
    form = (n >= min_players) & (share >= min_share) & ball_ok & (f["half"] > 0)
    # smooth: a formation lasts seconds, not frames
    form = roll_mean(form.astype(float), FPS // 2) > 0.8
    return form


def spells(mask, min_s=2.0):
    m = np.concatenate(([0], mask.astype(np.int8), [0])); d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (e - s) >= min_s * FPS
    return s[keep], e[keep]


def first_kickoffs(f, form=None, search_s=240):
    """Per half: the frame the ball first leaves the centre spot after an opening formation spell.
    Returns {half: frame or None}. Everything in the half before it is dead."""
    form = formation(f) if form is None else form
    B = load_atd()["ball"]; bm = B[:, :2] * PITCH
    bdist = np.linalg.norm(bm - PITCH / 2, axis=1)
    out = {}
    for h in (1, 2):
        idx = np.where(f["half"] == h)[0]; a = idx[0]
        s, e = spells(form[idx], min_s=2.0)
        s, e = s + a, e + a
        cand = s[s < a + search_s * FPS]
        if not len(cand):
            out[h] = None; continue
        t0 = cand[0]
        # the kick: ball tracked near the spot, then moving away
        win = np.arange(t0, min(t0 + 120 * FPS, idx[-1]))
        mv = np.isfinite(bdist[win]) & (bdist[win] < 12) & (np.nan_to_num(f["ball_speed"][win]) > 2.0)
        prior_spot = np.isfinite(bdist[win]) & (bdist[win] < 2.0)
        first_spot = np.argmax(prior_spot) if prior_spot.any() else 0
        mv[:first_spot] = False
        out[h] = int(win[np.argmax(mv)]) if mv.any() else int(e[0])
    return out
