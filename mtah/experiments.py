"""Automated experiment runner.

    .venv/bin/python -m mtah.experiments            # run every registered experiment + baselines
    .venv/bin/python -m mtah.experiments kick tau   # only experiments whose name contains a token

Each run: pipeline -> CSV in evaluation/results/exp/ -> format validation -> label-free diagnostics
-> public-window score (reference only, NEVER used to choose) -> one row in
eda/out/experiment_table.csv.
"""
import json, os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah import pipeline  # noqa: E402
from mtah.diagnostics import diagnose  # noqa: E402
from mtah.features import load as load_features  # noqa: E402
from mtah import holdout  # noqa: E402
from mtah.atd import ball_mode  # noqa: E402
SUF = ""

SEEDS = (0, 1, 2, 3, 4)
_F = load_features(raw=True); _MASKS = [holdout.masks(_F, s) for s in SEEDS]


def holdout_scores(cfg):
    res = []
    for hide, held in _MASKS:
        lab, _, post = pipeline.run(cfg, verbose=False, hide=hide, return_post=True)
        r = holdout.score(lab, held); r.update(holdout.nll(post, held)); res.append(r)
    return _agg(res)


def _agg(res):
    return {k: round(float(np.nanmean([r[k] for r in res])), 4) for k in res[0]}

OUT = os.path.join(ROOT, "evaluation", "results", "exp")
TABLE = os.path.join(ROOT, "eda", "out", "experiment_table.csv")
GT = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))

# (name, overrides, hypothesis)
REGISTRY = [
    ("v3_ref", dict(kickoff=0, presence_channel=0, control_dist=1.5, restarts=0, self_train=0), "reference: v3 as built yesterday"),
    ("v4_kickoff", dict(presence_channel=0, control_dist=1.5, restarts=0, self_train=0), "pre-kickoff and kickoff-formation frames are dead"),
    # touches
    ("touch_x3", dict(touch_gain=3.0), "touches carry more possession evidence than dribbling frames"),
    ("touch_x3_dv6", dict(touch_gain=3.0, touch_dv=6.0), "stricter touch definition"),
    # ablations
    ("abl_no_shape", dict(w_shape=0.0), "ablation: team-shape evidence removed"),
    ("abl_no_ball", dict(w_ball=0.0), "ablation: ball evidence removed"),
    ("abl_no_gapmodel", dict(gap_model=0), "ablation: gap-level dead-ball mixture removed"),
    ("abl_presence", dict(presence_channel=1), "ablation: ball presence counted as liveness evidence"),
    ("abl_no_control_gate", dict(control_dist=50.0, control_speed=99.0), "ablation: ball evidence on every visible frame"),
    # sensitivity of a-priori knobs (reported, not selected on the public window)
    ("sens_tau_ball_2.5", dict(tau_ball=2.5), "sensitivity"), ("sens_tau_ball_10", dict(tau_ball=10.0), "sensitivity"),
    ("sens_tau_shape_25", dict(tau_shape=25.0), "sensitivity"), ("sens_tau_shape_100", dict(tau_shape=100.0), "sensitivity"),
    ("sens_tau_live_12", dict(tau_live=12.5), "sensitivity"), ("sens_tau_live_50", dict(tau_live=50.0), "sensitivity"),
    ("sens_w_gap_0.5", dict(w_gap=0.5), "sensitivity"), ("sens_w_gap_2", dict(w_gap=2.0), "sensitivity"),
    ("sens_ctrl_1.0", dict(control_dist=1.0), "sensitivity"), ("sens_ctrl_2.0", dict(control_dist=2.0), "sensitivity"),
    # responsiveness grid (selected on held-out touches + support, never on the public window)
    ("resp_tb2.5_ctrl3", dict(tau_ball=2.5, control_dist=3.0, control_speed=15.0), "responsiveness"),
    ("resp_tb2.5_nogate", dict(tau_ball=2.5, control_dist=50.0, control_speed=99.0), "responsiveness"),
    ("resp_tb1_ctrl1.5", dict(tau_ball=1.0), "responsiveness"),
    ("resp_tb1_ctrl3", dict(tau_ball=1.0, control_dist=3.0, control_speed=15.0), "responsiveness"),
    ("resp_tb5_nogate", dict(control_dist=50.0, control_speed=99.0), "responsiveness"),
    ("resp_tb2.5_touch3", dict(tau_ball=2.5, touch_gain=3.0), "responsiveness"),
    ("resp_tb2.5_ctrl3_touch3", dict(tau_ball=2.5, control_dist=3.0, control_speed=15.0, touch_gain=3.0), "responsiveness"),
    # combinations around the presence channel
    ("combo_pres_touch3", dict(presence_channel=1, touch_gain=3.0), "presence + touch weighting"),
    ("combo_pres_tb2.5", dict(presence_channel=1, tau_ball=2.5), "presence + stronger ball evidence"),
    ("combo_pres_nogate", dict(presence_channel=1, control_dist=50.0, control_speed=99.0), "presence + ungated ball"),
    ("combo_pres_ctrl2", dict(presence_channel=1, control_dist=2.0), "presence + wider control gate"),
    ("combo_pres_ctrl2_touch3", dict(presence_channel=1, control_dist=2.0, touch_gain=3.0), "presence + wider gate + touches"),
    ("combo_pres_wgap2", dict(presence_channel=1, w_gap=2.0), "presence + stronger gap model (restart recall)"),
    # v5 = presence + 2 m control gate (new default); self-training on top of v5 and of the B variant
    ("v5", dict(restarts=0, self_train=0), "v5: presence channel + 2 m control gate"),
    ("v5_st1", dict(self_train=1), "self-training: shape model refit on confident decoded frames"),
    ("v5_st2", dict(self_train=2), "two rounds of self-training"),
    ("v5_st1_conf0.97", dict(self_train=1, st_conf=0.97), "self-training, stricter confidence"),
    ("B_nogate", dict(control_dist=50.0, control_speed=99.0), "variant B: presence + ungated ball"),
    ("B_nogate_st1", dict(control_dist=50.0, control_speed=99.0, self_train=1), "variant B + self-training"),
    ("v6_restarts", dict(restarts=1), "throw-in and goal-kick formations are dead"),
    ("v6_restarts_st1", dict(restarts=1, self_train=1), "restarts + self-training"),
    ("B_restarts", dict(restarts=1, control_dist=50.0, control_speed=99.0), "variant B + restarts"),
    ("v6", dict(), "v6 default"),
    ("v7_fill_team", dict(ball_fill=1), "broadcast-borrowed ball used for team evidence only"),
    ("c_2_1", dict(conf_gain=2.0), "confirmed touches x2"),
    ("c_3_1", dict(conf_gain=3.0), "confirmed touches x3"),
    ("c_2_05", dict(conf_gain=2.0, unconf_gain=0.5), "confirmed x2, stand-alone x0.5"),
    ("c_3_05", dict(conf_gain=3.0, unconf_gain=0.5), "confirmed x3, stand-alone x0.5"),
    ("c_5_05", dict(conf_gain=5.0, unconf_gain=0.5), "confirmed x5, stand-alone x0.5"),
    ("c_1_05", dict(unconf_gain=0.5), "stand-alone x0.5 only"),
    ("g_tb2.5_ts25.0", dict(tau_ball=2.5, tau_shape=25.0), "evidence-weight grid"),
    ("g_tb2.5_ts50.0", dict(tau_ball=2.5, tau_shape=50.0), "evidence-weight grid"),
    ("g_tb2.5_ts100.0", dict(tau_ball=2.5, tau_shape=100.0), "evidence-weight grid"),
    ("g_tb5.0_ts25.0", dict(tau_ball=5.0, tau_shape=25.0), "evidence-weight grid"),
    ("g_tb5.0_ts100.0", dict(tau_ball=5.0, tau_shape=100.0), "evidence-weight grid"),
    ("g_tb10.0_ts25.0", dict(tau_ball=10.0, tau_shape=25.0), "evidence-weight grid"),
    ("g_tb10.0_ts50.0", dict(tau_ball=10.0, tau_shape=50.0), "evidence-weight grid"),
    ("g_tb10.0_ts100.0", dict(tau_ball=10.0, tau_shape=100.0), "evidence-weight grid"),
    ("g_tb1.5_ts50.0", dict(tau_ball=1.5), "evidence-weight grid"),
    ("g_tb1.0_ts50.0", dict(tau_ball=1.0), "evidence-weight grid"),
    ("g_tb3.5_ts50.0", dict(tau_ball=3.5), "evidence-weight grid"),
    ("g_tb5.0_ts50.0", dict(), "evidence-weight grid (v7 default)"),
    ("t_v8", dict(), "round 4 reference: v8"),
    ("t_lead1.2", dict(tempo_lead_s=1.2), "tempo read 1.2 s ahead"),
    ("t_snapteam", dict(snap_team=1), "possession changes moved back to first arrival"),
    ("t_lead_snapteam", dict(tempo_lead_s=1.2, snap_team=1), "both"),
    ("t_all", dict(tempo_lead_s=1.2, snap_team=1, snap_dead=1), "both + dead ball starts when the ball left the field"),
    ("r3_v6", dict(ball_fill=0), "round 3 reference: v6"), ("r3_v7", dict(), "round 3 reference: v7"),
    ("r3_v4", dict(presence_channel=0, control_dist=1.5, restarts=0, self_train=0, ball_fill=0), "round 3 reference: v4"),
    ("v7_fill_all", dict(ball_fill=2), "broadcast-borrowed ball used for team and liveness evidence"),
    ("sens_shape_smooth_1", dict(shape_smooth_s=1.0), "sensitivity"), ("sens_shape_smooth_4", dict(shape_smooth_s=4.0), "sensitivity"),
]


def public_score(labels):
    table = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    r = pm.evaluate(GT, pm.label_codes(table), bootstrap_n=300)
    return {"pub_transF1_1s": round(r.get("trans_f1_25", float("nan")), 3), "pub_macroF1": round(r["macro_f1"], 4), "pub_spellF1": round(r["spell_f1_mean"], 4),
            "pub_lo": round(r["spell_bootstrap_low"], 3), "pub_hi": round(r["spell_bootstrap_high"], 3)}


def run_one(name, overrides, hypothesis):
    cfg = dict(pipeline.DEFAULTS); cfg.update(overrides)
    t0 = time.time(); labels, info = pipeline.run(cfg, verbose=False)
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, name + ".csv")
    with open(path, "w") as fh:
        fh.write("frame,label\n"); fh.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    reason = pm.validate_submission(pm.read_label_csv(path), len(GT))
    row = {"name": name, "hypothesis": hypothesis, "overrides": json.dumps(overrides), "valid": reason is None}
    row.update(diagnose(labels)); row.update(holdout_scores(cfg)); row.update(public_score(labels)); row["secs"] = round(time.time() - t0, 1)
    return row


def baselines():
    rows = []
    for b in ("baseline_smooth50", "baseline_smooth125"):
        lab = pd.read_csv(os.path.join(ROOT, "evaluation", "results", b + ".csv"))["label"].values.astype(str)
        row = {"name": b, "hypothesis": "organisers' baseline", "overrides": "", "valid": True}
        row.update(diagnose(lab)); row.update(public_score(lab)); rows.append(row)
    # baseline holdout: majority vote of nearest-player with the held-out touches removed
    from starter.baseline import smooth  # noqa
    near = np.where(np.isnan(_F["d_home"]) | np.isnan(_F["d_away"]), -1, np.where(_F["d_home"] < _F["d_away"], 0, 1)).astype(np.int8)
    for row, w in zip(rows, (50, 125)):
        sc = []
        for hide, held in _MASKS:
            c = near.copy(); c[hide] = -1; sc.append(holdout.score(smooth(c, w), held))
        row.update(_agg(sc))
    return rows


def save(rows):
    df = pd.DataFrame(rows)
    if os.path.isfile(TABLE):
        old = pd.read_csv(TABLE); df = pd.concat([old[~old.name.isin(df.name)], df], ignore_index=True)
    df.to_csv(TABLE, index=False)
    return df


def main(tokens, budget_s=85, skip_done=True):
    done = set(pd.read_csv(TABLE).name) if (skip_done and os.path.isfile(TABLE)) else set()
    todo = [(n + SUF, o, h) for n, o, h in REGISTRY if (not tokens or any(t in n for t in tokens)) and n + SUF not in done]
    t0 = time.time(); rows = []
    if not any(n.startswith("baseline") for n in done):
        rows += baselines(); save(rows)
    for name, ov, hyp in todo:
        if time.time() - t0 > budget_s:
            print("budget reached; rerun to continue", flush=True); break
        print("running", name, flush=True); rows.append(run_one(name, ov, hyp)); df = save(rows)
    df = pd.read_csv(TABLE)
    df["touch_f1"] = (2 * df.supported * df.ho_turnover_acc / (df.supported + df.ho_turnover_acc)).round(3)
    df.to_csv(TABLE, index=False)
    print("remaining:", len([r for r in todo if r[0] not in set(df.name)]))
    cols = ["name", "possessions", "mean_poss_s", "poss_lt2s", "dead_spells", "dead_share", "restart_recall",
            "supported", "ho_acc", "ho_sust_acc", "ho_isol_flip", "touch_f1", "pub_macroF1", "pub_spellF1", "pub_lo", "pub_hi", "valid"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
