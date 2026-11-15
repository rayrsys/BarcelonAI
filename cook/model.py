"""Supervised possession classifier with explicit control memory and sequence decoding."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from numba import njit
from sklearn.ensemble import HistGradientBoostingClassifier
from mtah.features import FPS, roll_mean

def shift(x, offset, half):
    ix = np.clip(np.arange(len(x)) + offset, 0, len(x) - 1)
    return np.where(half[ix] == half, x[ix], np.nan)

def design(f):
    names = sorted(k for k in f if k not in ('half', 'home_dir'))
    cols = [f[k].astype(float) for k in names]
    half = f['half']; bp = f['ball_present'] == 1
    dh, da = f['d_home'], f['d_away']; margin = da - dh
    extra = {'distance_margin': np.clip(margin, -20, 20),
             'distance_min': np.fmin(dh, da),
             'near_home': np.where(bp, np.sign(margin), np.nan)}
    control = bp & (np.fmin(dh, da) < 2.0) & (np.abs(margin) > 1.5) & (f['ball_edge_m'] > .3)
    control &= np.nan_to_num(f['ball_speed'], nan=99) < 12
    n = len(half); ix = np.arange(n)
    last = np.maximum.accumulate(np.where(control, ix, -n))
    nxt = np.minimum.accumulate(np.where(control, ix, 2*n)[::-1])[::-1]
    for key, where in (('last', last), ('next', nxt)):
        safe = np.clip(where, 0, n-1)
        good = (where >= 0) & (where < n) & (half[safe] == half)
        extra[key + '_control_age'] = np.where(good, np.minimum(np.abs(ix - where)/FPS, 120), 120)
        extra[key + '_control_team'] = np.where(good, np.sign(margin[safe]), 0)
    extra['control_agreement'] = extra['last_control_team'] * extra['next_control_team']
    for k, v in extra.items():
        names.append(k); cols.append(v)
    dynamic = ['ball_present', 'ball_speed', 'd_home', 'd_away', 'ball_edge_m',
               'speed_mean_1s', 'n_running', 'h_vx', 'a_vx', 'h_cx', 'a_cx']
    for k in dynamic:
        for off in (-75, -25, 25, 75):
            names.append(f'{k}_offset{off}'); cols.append(shift(f[k], off, half))
    for k in ('ball_present', 'ball_llr_home', 'speed_mean', 'h_vx', 'a_vx'):
        for w in (25, 75):
            names.append(f'{k}_mean{w}'); cols.append(roll_mean(f[k], w))
    X = np.column_stack(cols).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, names

def make_model(leaves=31):
    return HistGradientBoostingClassifier(max_iter=220, max_leaf_nodes=leaves,
        max_depth=None, learning_rate=.06, l2_regularization=8., min_samples_leaf=60,
        early_stopping=False, random_state=2027)

def swap_features(f):
    """Exchange the two teams without changing play or physical pitch geometry."""
    out = dict(f)
    for k in f:
        if k.startswith('h_') and 'a_' + k[2:] in f:
            out[k], out['a_' + k[2:]] = f['a_' + k[2:]], f[k]
    out['d_home'], out['d_away'] = f['d_away'], f['d_home']
    out['ball_llr_home'] = -f['ball_llr_home']
    out['ball_x_home'] = 1 - f['ball_x_home']
    out['home_dir'] = -f['home_dir']
    return out

@njit(cache=True)
def viterbi(e, logT):
    n, k = e.shape
    back = np.zeros((n, k), np.int8)
    delta = e[0].copy()
    for t in range(1, n):
        new = np.empty(k)
        for j in range(k):
            best, arg = -1e300, 0
            for i in range(k):
                val = delta[i] + logT[i,j]
                if val > best:
                    best, arg = val, i
            new[j] = best + e[t,j]; back[t,j] = arg
        delta = new
    out = np.empty(n, np.int8); out[-1] = np.argmax(delta)
    for t in range(n-1,0,-1):
        out[t-1] = back[t,out[t]]
    return out

def transitions(targets, halves):
    c = np.ones((3,3), float)
    for y, h in zip(targets, halves):
        ok = (y[:-1] >= 0) & (y[1:] >= 0) & (h[:-1] == h[1:]) & (h[1:] > 0)
        np.add.at(c, (y[:-1][ok], y[1:][ok]), 1)
    # H/A exchange symmetry prevents learning a home-team-specific possession prior.
    perm = np.array([1,0,2]); c = (c + c[perm][:,perm]) / 2
    return c / c.sum(1,keepdims=True)

def decode(p, half, T, evidence=1.0):
    labels = np.full(len(half), 'X', '<U1')
    logT = np.log(np.maximum(T, 1e-9))
    for h in (1,2):
        ix = np.flatnonzero(half == h)
        labels[ix] = np.array(['H','A','D'])[viterbi(np.log(np.clip(p[ix],1e-5,1))*evidence, logT)]
    return labels

def probabilities(model, f, stride=5):
    X, _ = design(f)
    ix = np.arange(0, len(X), stride)
    pred = model.predict_proba(X[ix])
    p = np.column_stack([np.interp(np.arange(len(X)), ix, pred[:,j]) for j in range(3)])
    return p

def score(y, labels, bootstrap=0):
    from evaluation import possession_metrics as pm
    truth = np.array(['X','H','A','D'])[y+1]
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels), bootstrap_n=bootstrap)
    keys = ['macro_f1','spell_f1_mean','n_gt_spells','n_pred_spells',
            'spell_bootstrap_low','spell_bootstrap_high','team_acc_gt_alive','alive_dead_f1']
    return {k: r[k] for k in keys}
