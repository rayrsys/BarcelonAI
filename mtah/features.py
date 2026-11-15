"""Frame-level features for the tactical source, indexed by video frame. Cached to data/cache.

Three families:
  ball   -- presence, gap length, speed, out-of-bounds, soft nearest-team evidence
  tempo  -- how fast the 22 players are moving (stoppage detector)
  shape  -- team geometry oriented by attacking direction (possession evidence without a ball)
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from mtah.atd import load as load_atd, PITCH, CACHE, ball_mode  # noqa: E402

FPS = 25


def roll_mean(x, w):
    """Centred NaN-aware rolling mean over +/- w frames."""
    v = np.where(np.isfinite(x), x, 0.0); c = np.isfinite(x).astype(float)
    V = np.concatenate(([0], np.cumsum(v))); C = np.concatenate(([0], np.cumsum(c)))
    i = np.arange(len(x)); lo = np.maximum(0, i - w); hi = np.minimum(len(x), i + w + 1)
    n = C[hi] - C[lo]
    return np.where(n > 0, (V[hi] - V[lo]) / np.maximum(n, 1), np.nan)


def run_lengths(mask):
    """For each frame: length of the True-run it sits in (0 where False) and position within it."""
    m = np.concatenate(([0], mask.astype(np.int8), [0])); d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    L = np.zeros(len(mask)); pos = np.zeros(len(mask))
    for a, b in zip(s, e):
        L[a:b] = b - a; pos[a:b] = np.arange(b - a)
    return L, pos


def half_and_dir(n):
    """half id per frame (0 outside graded halves) and home attacking direction (+1 = towards x=1)."""
    half = np.zeros(n, np.int8)
    for h, a, b in half_video_blocks(GAME):
        half[a:b + 1] = h
    return half


def build(raw=False):
    d = load_atd(raw=raw); P, B, team = d["players"], d["ball"], d["team"]
    n = len(B); f = {}
    half = half_and_dir(n); f["half"] = half
    # attacking direction learned from the data: the team whose players sit at lower mean x
    # in a half defends x=0 and attacks +x.
    home_dir = np.zeros(n, np.int8)
    for h in (1, 2):
        m = half == h
        hx = np.nanmean(P[m][:, team == 0, 0]); ax = np.nanmean(P[m][:, team == 1, 0])
        home_dir[m] = 1 if hx < ax else -1
    f["home_dir"] = home_dir

    # ---- ball ----
    bp = np.isfinite(B[:, 0]); f["ball_present"] = bp.astype(np.int8)
    L, pos = run_lengths(~bp)
    f["gap_len_s"] = L / FPS; f["gap_pos_s"] = pos / FPS
    bm = B[:, :2] * PITCH
    step = np.full(n, np.nan); step[1:] = np.linalg.norm(np.diff(bm, axis=0), axis=1) * FPS
    f["ball_speed"] = roll_mean(np.minimum(step, 40), 5)             # m/s, +/-0.2 s
    disp = np.full(n, np.nan); disp[FPS:] = np.linalg.norm(bm[FPS:] - bm[:-FPS], axis=1)
    f["ball_disp_1s"] = disp
    # ball velocity change across +/-3 frames: a touch, deflection or reception redirects the ball
    v = np.full((n, 2), np.nan); v[1:] = np.diff(bm, axis=0) * FPS
    vs = np.column_stack([roll_mean(v[:, 0], 1), roll_mean(v[:, 1], 1)])
    dv = np.full(n, np.nan); dv[3:-3] = np.linalg.norm(vs[6:] - vs[:-6], axis=1)
    f["ball_dv"] = np.minimum(dv, 60)
    x, y = B[:, 0], B[:, 1]
    f["ball_oob"] = (bp & ((x < 0) | (x > 1) | (y < 0) | (y > 1))).astype(np.int8)
    f["ball_edge_m"] = np.fmin(np.fmin(x, 1 - x) * PITCH[0], np.fmin(y, 1 - y) * PITCH[1])
    # ball x oriented to home attack, 0 = home goal, 1 = away goal
    f["ball_x_home"] = np.where(home_dir > 0, x, 1 - x)

    pm = P[:, :, :2] * PITCH
    dist = np.linalg.norm(pm - bm[:, None, :], axis=2)              # NaN where either missing
    ish = team[None] == 0
    dh = np.nanmin(np.where(ish, dist, np.nan), 1); da = np.nanmin(np.where(~ish, dist, np.nan), 1)
    f["d_home"], f["d_away"] = dh, da
    sig = 1.5
    kh = np.nansum(np.where(ish, np.exp(-0.5 * (dist / sig) ** 2), 0), 1)
    ka = np.nansum(np.where(~ish, np.exp(-0.5 * (dist / sig) ** 2), 0), 1)
    f["ball_llr_home"] = np.where(bp, np.log((kh + 0.02) / (ka + 0.02)), np.nan)  # >0 favours home

    # ---- tempo ----
    pp = np.isfinite(P[:, :, 0])
    spd = np.where(pp, P[:, :, 2], np.nan)
    f["n_players"] = pp.sum(1)
    ms = np.nanmean(spd, 1); f["speed_mean"] = ms
    f["speed_mean_1s"] = roll_mean(ms, FPS)
    f["speed_p80"] = np.nanpercentile(spd, 80, axis=1)
    f["n_running"] = np.nansum(spd > 4.0, 1)

    # ---- shape, oriented so +x = towards the opponent goal for each team ----
    k = 5  # displacement horizon in frames for velocities
    disp_x = np.full((n, P.shape[1]), np.nan); disp_x[k:] = (pm[k:, :, 0] - pm[:-k, :, 0]) * FPS / k
    for name, tm, sgn in (("h", 0, home_dir), ("a", 1, -home_dir)):
        sel = team == tm
        X = pm[:, sel, 0]; Y = pm[:, sel, 1]
        Xo = np.where(sgn[:, None] > 0, X, PITCH[0] - X)              # metres from own goal line
        f[f"{name}_cx"] = np.nanmean(Xo, 1)
        f[f"{name}_sx"] = np.nanstd(Xo, 1)
        f[f"{name}_sy"] = np.nanstd(Y, 1)
        srt = np.sort(np.where(np.isfinite(Xo), Xo, np.inf), 1)       # ascending, NaN last
        f[f"{name}_line"] = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan)  # 2nd deepest = back line
        f[f"{name}_front"] = np.nanmax(Xo, 1)
        f[f"{name}_vx"] = np.nanmedian(disp_x[:, sel] * sgn[:, None], 1)  # forward velocity
        f[f"{name}_in_opp_half"] = np.nansum(Xo > PITCH[0] / 2, 1)
    # --- extra shape geometry (v2) ---
    for name in ("h", "a"):
        f[f"{name}_area"] = f[f"{name}_sx"] * f[f"{name}_sy"]
        f[f"{name}_depth"] = f[f"{name}_front"] - f[f"{name}_line"]
    # overlap of the two blocks along the pitch (both measured from their own goal)
    f["overlap"] = (f["h_cx"] + f["a_cx"]) - PITCH[0]
    # pressing: mean distance from each player to the nearest opponent
    ish = team == 0
    Ph, Pa = pm[:, ish, :], pm[:, ~ish, :]
    dd = np.linalg.norm(Ph[:, :, None, :] - Pa[:, None, :, :], axis=3)   # n x 23 x 23 (slots)
    f["h_press"] = np.nanmean(np.nanmin(dd, axis=2), axis=1)               # home players' nearest away
    f["a_press"] = np.nanmean(np.nanmin(dd, axis=1), axis=1)
    return {k_: np.asarray(v) for k_, v in f.items()}


def load(rebuild=False, raw=None):
    if raw is None:
        raw = ball_mode() != "fill"
    path = os.path.join(CACHE, "features_tactical.npz" if raw else "features_tactical_fill.npz")
    if rebuild or not os.path.isfile(path):
        np.savez(path, **build(raw))
    z = np.load(path)
    return {k_: z[k_] for k_ in z.files}


if __name__ == "__main__":
    f = load(rebuild=True); load(rebuild=True, raw=True)
    print(len(f), "features:", ", ".join(sorted(f)))
