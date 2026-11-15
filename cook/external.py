"""Convert IDSSE XML to normalized tracking, retaining labels in a separate array.

Coordinates: centred metres -> normalized pitch. DFL speed: km/h -> m/s.
The competition game is never read here. Labels are used only by the training/evaluation driver.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dfl'

def parse(match):
    cache = DATA / f'{match}.npz'
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    meta = ET.parse(DATA / f'{match}_meta.xml').getroot()
    general = next(meta.iter('General'))
    home = general.get('HomeTeamId'); away = general.get('GuestTeamId')
    players, teams = [], []
    for t in meta.iter('Team'):
        for p in t.iter('Player'):
            players.append(p.get('PersonId')); teams.append(0 if t.get('TeamId') == home else 1)
    env = next(meta.iter('Environment'))
    pitch = np.array([float(env.get('PitchX')), float(env.get('PitchY'))], np.float32)
    blocks = {1: [], 2: []}
    # One FrameSet at a time bounds memory to one player-half.
    for _, el in ET.iterparse(DATA / f'{match}_tracking.xml', events=('end',)):
        if el.tag != 'FrameSet':
            continue
        sec = el.get('GameSection')
        if sec not in ('firstHalf', 'secondHalf'):
            el.clear(); continue
        half = 1 if sec == 'firstHalf' else 2
        pid, tid = el.get('PersonId'), el.get('TeamId')
        frames = list(el)
        is_ball = any(f.get('BallStatus') is not None for f in frames[:3])
        if not is_ball and pid not in players:
            el.clear(); continue
        ids = np.array([int(f.get('N')) for f in frames], np.int32)
        xyz = np.array([[float(f.get('X', 'nan')), float(f.get('Y', 'nan')),
                         float(f.get('S', 'nan')) / 3.6] for f in frames], np.float32)
        xyz[:, :2] = xyz[:, :2] / pitch + .5
        labels = None
        if is_ball:
            # H=0, A=1, D=2, X=-1. Unknown possession while live is ungraded.
            labels = np.array([2 if f.get('BallStatus') == '0' else
                               (int(f.get('BallPossession')) - 1 if f.get('BallPossession') in ('1', '2') else -1)
                               for f in frames], np.int8)
        blocks[half].append((pid, is_ball, ids, xyz, labels))
        el.clear()
    pp, bb, hh, yy = [], [], [], []
    for half, records in blocks.items():
        ball_records = [r for r in records if r[1]]
        if not ball_records:
            raise ValueError(f'No ball records for {match} half {half}')
        lo = min(r[2].min() for r in ball_records); hi = max(r[2].max() for r in ball_records)
        n = hi - lo + 1
        P = np.full((n, len(players), 3), np.nan, np.float32)
        B = np.full((n, 3), np.nan, np.float32); y = np.full(n, -1, np.int8)
        for pid, is_ball, ids, xyz, labels in records:
            ix = ids - lo; good = (ix >= 0) & (ix < n)
            if is_ball:
                B[ix[good]] = xyz[good]; y[ix[good]] = labels[good]
            else:
                P[ix[good], players.index(pid)] = xyz[good]
        pp.append(P); bb.append(B); hh.append(np.full(n, half, np.int8)); yy.append(y)
        # An explicit X frame prevents possession merging across halftime.
        if half == 1:
            pp.append(np.full((1, len(players), 3), np.nan, np.float32))
            bb.append(np.full((1, 3), np.nan, np.float32)); hh.append(np.zeros(1, np.int8)); yy.append(np.full(1, -1, np.int8))
    out = dict(players=np.concatenate(pp), ball=np.concatenate(bb), half=np.concatenate(hh),
               target=np.concatenate(yy), team=np.array(teams, np.int8), pitch=pitch)
    assert set(np.unique(out['target'])) <= {-1, 0, 1, 2}
    np.savez_compressed(cache, **out)
    return out

def corrupt(d, seed=2027):
    """Label-independent sensor stress test, fixed before looking at external scores.

    Alternate observed / missing periods drawn from the competition sensor's empirical run
    lengths. This transfers sensor statistics, never possession labels. Add moderate spatial
    noise and occasional player-track loss to avoid training on immaculate tracking alone.
    """
    from mtah.features import load
    f = load(raw=True); bp = f['ball_present'].astype(bool); graded = f['half'] > 0
    durations = {}
    for value in (False, True):
        mask = (bp == value) & graded
        delta = np.diff(np.r_[False, mask, False].astype(int))
        durations[value] = np.flatnonzero(delta == -1) - np.flatnonzero(delta == 1)
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in d.items()}
    # Targets deliberately do not enter the corruption process.
    P, B = out['players'], out['ball']; pitch = out['pitch']
    from scipy.ndimage import gaussian_filter1d
    # Smooth spatial noise: independent frame jitter would manufacture unrealistic velocity.
    for coords, sigma in ((P[:, :, :2], .20), (B[:, :2], .35)):
        noise = gaussian_filter1d(rng.normal(size=coords.shape).astype(np.float32), 5, axis=0)
        coords += noise * (sigma * np.sqrt(10 * np.sqrt(np.pi))) / pitch
    for h in (1, 2):
        idx = np.flatnonzero(d['half'] == h); pos = int(idx[0]); stop = int(idx[-1]) + 1
        visible = True
        while pos < stop:
            length = int(rng.choice(durations[visible]))
            if not visible:
                B[pos:min(pos + length, stop)] = np.nan
            pos += length; visible = not visible
        # Lose a random 10% of players in 2 s chunks. No identity features are ever used.
        for a in range(int(idx[0]), stop, 50):
            missing = rng.random(P.shape[1]) < .1
            P[a:min(a + 50, stop), missing] = np.nan
    return out

def features(d):
    """Reuse the prior v9 coordinate/geometry calculations on a different match."""
    from unittest.mock import patch
    from mtah import features as mf
    with patch.object(mf, 'load_atd', return_value=d), patch.object(mf, 'half_and_dir', return_value=d['half']), patch.object(mf, 'PITCH', d['pitch']):
        return mf.build(raw=True)

def prepare(match, seed=2027):
    path = DATA / f'{match}_features_{seed}.npz'
    d = parse(match)
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}, d['target']
    f = features(corrupt(d, seed))
    np.savez_compressed(path, **f)
    return f, d['target']

if __name__ == '__main__':
    import sys
    for match in sys.argv[1:]:
        f, y = prepare(match)
        print(match, 'frames', len(y), 'label counts', np.bincount(y + 1).tolist(), flush=True)
