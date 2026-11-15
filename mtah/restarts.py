"""Restart-formation detectors from the laws of the game, tracking only.

throw-in : the ball is held beyond a touchline (outside the field along the width)
goal kick: the ball is slow inside a six-yard box and no attacking player is in that penalty area
Both mark the ball as dead while the formation lasts.
"""
import numpy as np
from mtah.atd import load as load_atd, PITCH
from mtah.features import roll_mean, FPS


def detectors(f, out_m=0.3):
    d = load_atd(); P, B, team = d["players"], d["ball"], d["team"]
    bx, by = B[:, 0] * PITCH[0], B[:, 1] * PITCH[1]
    vis = np.isfinite(bx) & (f["half"] > 0)
    throw = vis & ((by < -out_m) | (by > PITCH[1] + out_m)) & (bx > 0) & (bx < PITCH[0])
    slow = np.nan_to_num(f["ball_speed"], nan=9) < 4.0
    cy = PITCH[1] / 2
    gk = np.zeros(len(bx), bool)
    X, Y = P[:, :, 0] * PITCH[0], P[:, :, 1] * PITCH[1]
    hd = f["home_dir"].astype(float)
    # the ball must recently have left the field over that goal line (shot or cross wide, or a
    # track gap that began with the ball at that line): otherwise a keeper holding it is live play
    n = len(bx); recent = {}
    for end in (0, 1):
        over = vis & ((bx < 0) if end == 0 else (bx > PITCH[0]))
        gap_start = np.zeros(n, bool); miss = ~np.isfinite(bx)
        st = np.where(np.diff(np.concatenate(([0], miss.astype(np.int8)))) == 1)[0]
        for s_ in st:
            j = s_ - 1
            if j >= 0 and np.isfinite(bx[j]) and ((bx[j] < 3.0) if end == 0 else (bx[j] > PITCH[0] - 3.0)):
                gap_start[s_] = True
        ev = (over | gap_start).astype(float)
        c = np.concatenate(([0], np.cumsum(ev))); i = np.arange(n); lo = np.maximum(0, i - 40 * FPS)
        recent[end] = (c[i + 1] - c[lo]) > 0
    for end in (0, 1):                                   # goal at x=0 or x=L
        six = vis & slow & recent[end] & (np.abs(by - cy) < 9.16 + 0.5) & ((bx < 5.5 + 0.5) if end == 0 else (bx > PITCH[0] - 6.0))
        # defending team of this end: home if home defends it (home_dir>0 means home defends x=0)
        defends_home = (hd > 0) if end == 0 else (hd < 0)
        attackers = np.where(defends_home[:, None], team[None] == 1, team[None] == 0)
        inbox = ((X < 16.5) if end == 0 else (X > PITCH[0] - 16.5)) & (np.abs(Y - cy) < 20.16)
        att_in = np.nansum(attackers & inbox & np.isfinite(X), 1)
        gk |= six & (att_in <= 2)          # law says 0; tracking noise near a crowded box
    sm = lambda m: roll_mean(m.astype(float), FPS // 2) > 0.6
    return {"throw": sm(throw), "goalkick": sm(gk)}


def spells(mask, min_s):
    m = np.concatenate(([0], mask.astype(np.int8), [0])); d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]; k = (e - s) >= min_s * FPS
    return s[k], e[k]
