"""Align the broadcast tracking to the tactical timeline and borrow its ball where tactical lost it.

The broadcast recording has its own clock, cuts and replays. Each 10 s broadcast chunk is slid
along the whole tactical ball track (masked sum of squared distances via FFT); the lag with the
smallest mean distance is that chunk's offset. Chunks are kept only when the fit is tight and
unambiguous. No labels, no video, no manual offsets.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.atd import load as load_atd, PITCH, CACHE  # noqa: E402

CHUNK = 250          # frames per broadcast chunk
MIN_OVERLAP = 100    # frames with the ball visible in both sources at the chosen lag
MAX_RMS = 2.0        # metres
MIN_RATIO = 1.5      # runner-up lag (>2 s away) must be this much worse


def _corr(a_fft, b, n):
    return np.fft.irfft(a_fft * np.conj(np.fft.rfft(b, n)), n)


def align(flip=False):
    T = load_atd("tactical", raw=True)["ball"][:, :2] * PITCH
    Bc = load_atd("broadcast")["ball"][:, :2] * PITCH
    if flip:
        Bc = PITCH - Bc
    nt, nb = len(T), len(Bc)
    n = 1 << int(np.ceil(np.log2(nt + CHUNK)))
    mt = np.isfinite(T[:, 0]).astype(float); T0 = np.nan_to_num(T)
    F = {"m": np.fft.rfft(mt, n), "x": np.fft.rfft(T0[:, 0] * mt, n), "y": np.fft.rfft(T0[:, 1] * mt, n),
         "s": np.fft.rfft((T0 ** 2).sum(1) * mt, n)}
    rows = []
    for a in range(0, nb - CHUNK + 1, CHUNK):
        c = Bc[a:a + CHUNK]; mb = np.isfinite(c[:, 0]).astype(float)
        if mb.sum() < MIN_OVERLAP:
            continue
        c0 = np.nan_to_num(c)
        cnt = _corr(F["m"], mb, n)[:nt]
        ssd = (_corr(F["s"], mb, n) - 2 * _corr(F["x"], c0[:, 0] * mb, n) - 2 * _corr(F["y"], c0[:, 1] * mb, n)
               + _corr(F["m"], (c0 ** 2).sum(1) * mb, n))[:nt]
        ok = cnt >= MIN_OVERLAP - 0.5
        if not ok.any():
            continue
        msd = np.where(ok, np.maximum(ssd, 0) / np.maximum(cnt, 1), np.inf)
        k = int(np.argmin(msd)); best = msd[k]
        far = msd.copy(); far[max(k - 50, 0):k + 51] = np.inf
        second = far.min()
        rows.append((a, k - a, np.sqrt(best), np.sqrt(second) / max(np.sqrt(best), 1e-6), cnt[k]))
    return np.array(rows)   # broadcast start, offset (tactical = broadcast + offset), rms, ratio, overlap


def accepted(rows):
    return rows[(rows[:, 2] < MAX_RMS) & (rows[:, 3] > MIN_RATIO)]


def player_cost(Pt, tt, Pb, tb, a, off, step=10):
    """Median distance from each broadcast player to the nearest same-team tactical player."""
    d = []
    for i in range(a, a + CHUNK, step):
        t = i + off
        if t < 0 or t >= len(Pt):
            continue
        for tm in (0, 1):
            pb = Pb[i][tb == tm]; pb = pb[np.isfinite(pb[:, 0])]
            pt = Pt[t][tt == tm]; pt = pt[np.isfinite(pt[:, 0])]
            if len(pb) and len(pt):
                d.append(np.min(np.linalg.norm(pb[:, None] - pt[None], axis=2), axis=1))
    return (np.median(np.concatenate(d)), sum(len(x) for x in d)) if d else (np.inf, 0)


def offsets(max_player_m=1.0, min_players=60):
    """Per broadcast chunk: verified offset or None. Ball-aligned chunks seed the candidate set;
    every chunk (seed or not) is then verified against player positions at the candidate offsets."""
    rows = accepted(align())
    cands = np.unique(rows[:, 1]).astype(int)
    dt, db = load_atd("tactical", raw=True), load_atd("broadcast")
    Pt, Pb = dt["players"][:, :, :2] * PITCH, db["players"][:, :, :2] * PITCH
    out = {}
    seed = {int(a): int(o) for a, o, *_ in rows}
    for a in range(0, len(Pb) - CHUNK + 1, CHUNK):
        order = sorted(cands, key=lambda o: abs(o - seed.get(a, o)))    # the chunk's own seed first
        near = [o for o in order if any(abs((a2 + o2) - (a + o)) < 3000 and o2 == o for a2, o2 in seed.items())]
        best = (np.inf, None)
        for o in ([seed[a]] if a in seed else []) + near:
            c, n = player_cost(Pt, dt["team"], Pb, db["team"], a, int(o))
            if n >= min_players and c < best[0]:
                best = (c, int(o))
        if best[0] < max_player_m:
            out[a] = best[1]
    return out


def fill(rebuild=False):
    """Tactical-indexed ball (n,2 normalised) borrowed from broadcast; NaN where nothing to borrow."""
    path = os.path.join(CACHE, "ball_fill.npy")
    if not rebuild and os.path.isfile(path):
        return np.load(path)
    Bc = load_atd("broadcast")["ball"][:, :2]; nt = len(load_atd("tactical", raw=True)["ball"])
    out = np.full((nt, 2), np.nan, np.float32)
    for a, off in sorted(offsets().items()):
        t = np.arange(a, a + CHUNK) + off; ok = (t >= 0) & (t < nt)
        out[t[ok]] = Bc[a:a + CHUNK][ok]
    np.save(path, out)
    return out
