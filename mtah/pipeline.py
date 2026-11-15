"""End-to-end: provided data -> possession CSV.

    .venv/bin/python -m mtah.pipeline --name v1

Every knob below is set a priori and documented; none is selected on the public window.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.features import load as load_features, roll_mean, FPS  # noqa: E402
from mtah.shape_model import load_oof, pseudo_labels  # noqa: E402
from mtah import hmm  # noqa: E402
from mtah.gaps import dead_prob  # noqa: E402
from mtah.kickoff import formation, first_kickoffs  # noqa: E402
from mtah.restarts import detectors as restart_detectors  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "evaluation", "results")
LOG = os.path.join(ROOT, "eda", "out", "experiments.jsonl")

DEFAULTS = dict(
    tau_ball=2.5,     # fitted by held-out touch likelihood (elbow; see experiment table g_tb*)
    tau_shape=50.0,   # shape evidence is smoothed over +/-2 s -> ~one independent obs per 2 s
    tau_live=25.0,    # same for liveness
    w_ball=1.0,       # weight of soft-nearest-player ball evidence
    w_shape=1.0,      # weight of shape-model evidence
    shape_smooth_s=2.0,
    shape_version=1,
    tempo_lead_s=1.2,   # read player tempo this far ahead: players react ~1.2 s after the ball state changes (eda/out/tempo_lag.json)
    snap_team=1,        # 1: move each change of possession back to when the new team first became nearest to the ball
    snap_dead=0,        # 1: start a dead ball when the ball left the field, if that happened just before
    snap_window_s=3.0,
    ball_fill=1,        # 0: provided ball only; 1: broadcast-borrowed ball for team evidence only; 2: for everything
    ball_clip=4.0,
    control_dist=2.0,   # m: ball this close to a player can be under control (chosen on held-out touches)
    control_speed=8.0,  # m/s: faster than this the ball is travelling, not controlled
    presence_channel=1, # 1: ball presence counts as liveness evidence (chosen on held-out touches, exp table)
    gap_model=1,        # 1: gap-level mixture decides dead/alive inside ball gaps
    w_gap=1.0,          # nats of evidence per second of gap, per unit logit of the gap's dead prob
    conf_gain=1.0,      # multiplier on ball evidence for touches confirmed by the same team's next touch (or held)
    unconf_gain=1.0,    # multiplier for stand-alone touches
    conf_window_s=4.0, conf_hold_s=1.0,
    touch_gain=1.0,     # multiplier on ball evidence at touch frames (ball redirected near a player)
    touch_dv=4.0,       # m/s velocity change that counts as a touch
    kickoff=1,          # 1: everything in a half before its first kickoff is dead; kickoff formations are dead
    w_form=0.3,         # nats per frame of dead evidence inside a kickoff formation
    restarts=1,         # 1: throw-in and goal-kick formations are dead
    w_restart=0.3,      # nats per frame of dead evidence inside those formations
    self_train=1,       # rounds of shape-model self-training on the HMM's own confident frames
    st_conf=0.9,        # posterior needed for a frame to become a self-training label
    em_step=5,        # EM runs at 5 fps; decoding at full 25 fps
    em_iters=10,        # converges by ~6 (see logs)
)


def confirm_weights(control, sign, cfg):
    """Per-frame multiplier on ball evidence: a touch counts more when the same team has the
    neighbouring touch too (a controlled possession), less when it stands alone (a deflection)."""
    w = np.ones(len(control))
    idx = np.where(control)[0]
    if not len(idx) or (cfg["conf_gain"] == 1.0 and cfg["unconf_gain"] == 1.0):
        return w
    br = np.where((np.diff(idx) > 5) | (sign[idx[1:]] != sign[idx[:-1]]))[0]
    st = np.concatenate(([idx[0]], idx[br + 1])); en = np.concatenate((idx[br], [idx[-1]])) + 1
    tm = sign[st]; win = int(cfg["conf_window_s"] * FPS)
    for i in range(len(st)):
        nxt = i + 1 < len(st) and st[i + 1] - en[i] <= win and tm[i + 1] == tm[i]
        prv = i > 0 and st[i] - en[i - 1] <= win and tm[i - 1] == tm[i]
        held = en[i] - st[i] >= cfg["conf_hold_s"] * FPS
        w[st[i]:en[i]] = cfg["conf_gain"] if (nxt or held) else (1.0 if prv else cfg["unconf_gain"])
    return w


def team_evidence(f, cfg, hide=None):
    """(ball_llr, shape_llr): log-likelihood ratios for home vs away, >0 favours home."""
    dmin = np.fmin(f["d_home"], f["d_away"])
    control = (f["ball_present"] == 1) & (dmin < cfg["control_dist"]) & (np.nan_to_num(f["ball_speed"], nan=99) < cfg["control_speed"])
    ball = np.where(control, np.clip(np.nan_to_num(f["ball_llr_home"]), -cfg["ball_clip"], cfg["ball_clip"]), 0.0)
    touch = (f["ball_present"] == 1) & (dmin < cfg["control_dist"]) & (np.nan_to_num(f["ball_dv"]) > cfg["touch_dv"])
    ball = np.where(touch, ball * cfg["touch_gain"], ball)
    if hide is not None:
        ball = np.where(hide, 0.0, ball); control = control & ~hide
    ball = ball * confirm_weights(control, np.sign(ball), cfg)
    p = roll_mean(load_oof(version=cfg["shape_version"]), int(cfg["shape_smooth_s"] * FPS))
    ok, y = pseudo_labels(f, cfg["shape_version"]); prior = y[ok].mean()
    p = np.clip(np.nan_to_num(p, nan=prior), 0.02, 0.98)
    shape = np.log(p / (1 - p)) - np.log(prior / (1 - prior))  # posterior -> likelihood ratio
    return cfg["w_ball"] * ball, cfg["w_shape"] * shape


def liveness_obs(f, cfg):
    bp = f["ball_present"] == 1
    lead = int(round(cfg["tempo_lead_s"] * FPS)); sm = f["speed_mean_1s"]
    if lead:
        sm = np.concatenate((sm[lead:], np.full(lead, np.nan)))
    f = dict(f); f["speed_mean_1s"] = sm
    sp = np.log(np.nan_to_num(f["speed_mean_1s"], nan=1.5) + 0.05)
    bs = np.log(np.nan_to_num(f["ball_speed"], nan=1.0) + 0.1)
    obs = {
        "ball_present": (bp.astype(float), np.ones_like(bp), "bern"),
        "log_gap": (np.log(f["gap_len_s"] + 0.04), ~bp, "gauss"),
        "log_speed": (sp, np.isfinite(f["speed_mean_1s"]), "gauss"),
        "log_ball_speed": (bs, bp & np.isfinite(f["ball_speed"]), "gauss"),
        "oob": (f["ball_oob"].astype(float), bp, "bern"),
    }
    if not cfg["presence_channel"]:
        obs.pop("ball_present")
    if cfg["gap_model"]:
        obs.pop("log_gap")
    return obs


def halves(f):
    h = f["half"]; return [np.where(h == k)[0] for k in (1, 2)]


def block_sum(e, k):
    n = len(e) // k * k
    head = e[:n].reshape(-1, k, e.shape[1]).sum(1)
    if n < len(e):
        head = np.vstack([head, e[n:].sum(0, keepdims=True)])
    return head


def self_train_shape(f, gam, cfg, idx_all):
    """Retrain the shape model on frames the HMM is confident about (labels from our own decode).
    Out-of-fold over 5-minute blocks, ball-free features, every 5th frame to limit correlation."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from mtah.shape_model import design
    X, _ = design(f, 1)
    post = np.zeros((len(f["half"]), 3)); post[idx_all] = gam
    ph, pa = post[:, 0], post[:, 1]
    conf = ((ph > cfg["st_conf"]) | (pa > cfg["st_conf"])) & (f["half"] > 0)
    conf[np.arange(len(conf)) % 5 != 0] = False
    y = (ph > pa).astype(np.int8)
    blk = (np.arange(len(conf)) // (300 * FPS)) % 5
    p = np.full(len(conf), np.nan)
    for k_ in range(5):
        tr = conf & (blk != k_)
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        p[blk == k_] = m.predict_proba(X[blk == k_])[:, 1]
    prior = y[conf].mean()
    p = np.clip(np.nan_to_num(roll_mean(p, int(cfg["shape_smooth_s"] * FPS)), nan=prior), 0.02, 0.98)
    return cfg["w_shape"] * (np.log(p / (1 - p)) - np.log(prior / (1 - prior)))


def snap(labels, f, cfg):
    """Boundary timing from the laws of the game, applied after decoding."""
    if not (cfg["snap_team"] or cfg["snap_dead"]):
        return labels
    ff = load_features(raw=False) if cfg["ball_fill"] else f
    lab = labels.copy(); W = int(cfg["snap_window_s"] * FPS)
    dh, da = ff["d_home"], ff["d_away"]; seen = ff["ball_present"] == 1; oob = ff["ball_oob"] == 1
    ch = np.where(lab[1:] != lab[:-1])[0] + 1
    prev = 0
    for s_ in ch:
        old, new = lab[s_ - 1], lab[s_]
        lo = max(prev + 1, s_ - W); u = s_
        if cfg["snap_team"] and old in "HA" and new in "HA":
            near_new = seen & ((dh < da) if new == "H" else (da < dh))
            while u > lo and near_new[u - 1]:
                u -= 1
        elif cfg["snap_dead"] and old in "HA" and new == "D":
            w = np.where(oob[lo:s_])[0]
            if len(w):
                u = lo + int(w[0])
        if u < s_:
            lab[u:s_] = new
        prev = u
    return lab


def run(cfg, verbose=True, hide=None, return_post=False):
    os.environ["MTAH_BALL"] = "fill" if cfg["ball_fill"] == 2 else ""
    f = load_features()
    ball_llr, shape_llr = team_evidence(f, cfg, hide)
    if cfg["ball_fill"] == 1:
        ff = load_features(raw=False)
        fill_llr, _ = team_evidence(ff, cfg, hide)
        ball_llr = np.where(f["ball_present"] == 1, ball_llr, fill_llr)
    if hide is not None:                      # held-out touch test: remove ball evidence there
        ball_llr = np.where(hide, 0.0, ball_llr)
    labels, info, gam, idx_all = _decode(f, ball_llr, shape_llr, cfg, verbose)
    for r in range(cfg["self_train"]):
        shape_llr = self_train_shape(f, gam, cfg, idx_all)
        labels, info, gam, idx_all = _decode(f, ball_llr, shape_llr, cfg, verbose)
    labels = snap(labels, f, cfg)
    if return_post:
        post = np.full((len(labels), 3), np.nan); post[idx_all] = gam
        return labels, info, post
    return labels, info


def _decode(f, ball_llr, shape_llr, cfg, verbose):
    llr = ball_llr / cfg["tau_ball"] + shape_llr / cfg["tau_shape"]   # already tempered
    obs = liveness_obs(f, cfg)
    idx_all = np.concatenate(halves(f))
    sub = {k: (v[idx_all], m[idx_all], kind) for k, (v, m, kind) in obs.items()}
    live = hmm.Liveness(sub)
    spd = np.nan_to_num(f["speed_mean_1s"][idx_all], nan=1.5)
    dead0 = 1 / (1 + np.exp(-(3 * (1.3 - spd) + 1.0 * (f["gap_len_s"][idx_all] > 3) - 0.5)))
    live.init(dead0)
    k = cfg["em_step"]
    T = np.array([[1 - k / 300 - k / 750, k / 300, k / 750], [k / 300, 1 - k / 300 - k / 750, k / 750], [k / 750, k / 750, 1 - k / 375]])
    log_pi = np.log(np.full(3, 1 / 3))
    bounds = np.cumsum([0] + [len(i) for i in halves(f)])
    llr_s = llr[idx_all]
    extra_dead = np.zeros(len(idx_all))
    if cfg["gap_model"]:
        pg, _ = dead_prob(f); pg = np.clip(pg[idx_all], 0.01, 0.99)
        extra_dead = np.where(np.isfinite(pg), np.log(pg / (1 - pg)) * cfg["w_gap"] / FPS, 0.0)
    if cfg["kickoff"]:
        full = np.zeros(len(f["half"]))
        form = formation(f); full += np.where(form, cfg["w_form"], 0.0)
        for h, k_ in first_kickoffs(f, form).items():
            if k_ is not None:
                a_ = np.where(f["half"] == h)[0][0]; full[a_:k_] += 10.0
        extra_dead = extra_dead + full[idx_all]
    if cfg["restarts"]:
        det = restart_detectors(f)
        extra_dead = extra_dead + cfg["w_restart"] * (det["throw"] | det["goalkick"])[idx_all]
    def E_(ll):
        e_ = hmm.emissions(llr_s, ll, 1.0, cfg["tau_live"]); e_[:, 2] += extra_dead; return e_
    for it in range(cfg["em_iters"]):
        e = E_(live.loglik())
        gam_full = np.zeros((len(idx_all), 3)); xi = np.zeros((3, 3)); LL = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            eb = block_sum(e[a:b], k)
            g, x, ll = hmm.forward_backward(eb, np.log(T + hmm.LOG_EPS), log_pi)
            gam_full[a:b] = np.repeat(g, k, axis=0)[: b - a]; xi += x; LL += ll
        T = hmm.tie_transitions(xi); live.m_step(gam_full)
        if verbose:
            print(f"  em {it:2d}  ll {LL:12.1f}  p_ha/s {T[0,1]*25/k:.4f}  p_ad/s {T[0,2]*25/k:.4f}  p_da/s {T[2,0]*2*25/k:.4f}  dead {gam_full[:,2].mean():.3f}")
    T25 = hmm.rescale(T, k)
    e = E_(live.loglik())
    labels = np.full(len(llr), "X", dtype="<U1")
    for (a, b), idx in zip(zip(bounds[:-1], bounds[1:]), halves(f)):
        path = hmm.viterbi(e[a:b], np.log(T25 + hmm.LOG_EPS), log_pi)
        labels[idx] = np.array(["H", "A", "D"])[path]
    params = {n: {s: [round(float(x), 3) for x in np.atleast_1d(v)] for s, v in p.items()} for n, p in live.params.items()}
    return labels, {"T_per_s": {"h_to_a": T[0, 1] * 25 / k, "alive_to_dead": T[0, 2] * 25 / k, "dead_to_alive": T[2, 0] * 2 * 25 / k}, "liveness_params": params}, gam_full, idx_all


def write(labels, name):
    os.makedirs(RESULTS, exist_ok=True); path = os.path.join(RESULTS, name + ".csv")
    with open(path, "w") as fh:
        fh.write("frame,label\n"); fh.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    return path


def summarise(labels):
    g = labels[labels != "X"]
    ch = np.concatenate(([True], g[1:] != g[:-1])); segs = g[ch]
    lens = np.diff(np.concatenate((np.where(ch)[0], [len(g)])))
    poss = np.isin(segs, ["H", "A"])
    return {"n_possessions": int(poss.sum()), "mean_poss_s": round(float(lens[poss].mean() / FPS), 2),
            "n_dead_spells": int((segs == "D").sum()),
            "share": {c: round(float((g == c).mean()), 3) for c in "HAD"}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    for k_, v_ in DEFAULTS.items():
        ap.add_argument("--" + k_, type=type(v_), default=v_)
    args = vars(ap.parse_args()); name = args.pop("name")
    t0 = time.time(); labels, info = run(args)
    path = write(labels, name); s = summarise(labels)
    rec = {"name": name, "cfg": args, "summary": s, "model": info, "secs": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as fh: fh.write(json.dumps(rec) + "\n")
    print(json.dumps({"name": name, "summary": s, "T_per_s": info["T_per_s"]}, indent=1))
