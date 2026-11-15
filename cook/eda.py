"""Independent EDA: aggregates only, no ground-truth reads or data extracts."""
import json
from pathlib import Path
import numpy as np
from mtah.features import load, FPS
from mtah.shape_model import pseudo_labels
from mtah.holdout import clusters
from starter.load_smart_tagging import load as load_tags

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'cook' / 'reports'

def tag_features(n):
    events = load_tags()
    # These are the automatically generated phase definitions in OVERVIEW.md.
    positive = {'BUILD UP', 'PROGRESSION', 'CREATING CHANCES', 'ATTACKING TRANSITION',
                'LONG BALL', 'BALL IN FINAL THIRD', 'BALL IN THE BOX', 'PLAYERS IN THE BOX'}
    negative = {'DEFENDING IN ATTACKING THIRD', 'DEFENDING IN MIDDLE THIRD',
                'DEFENDING IN DEFENSIVE THIRD', 'DEFENSIVE TRANSITION'}
    codes = sorted({str(e['code']).strip().upper() for e in events})
    votes = np.zeros((n, len(codes)), np.float32)
    presence = np.zeros_like(votes)
    counts = {}
    for e in events:
        code = str(e['code']).strip().upper()
        counts[code] = counts.get(code, 0) + 1
        team = str(e['tags'].get('Team', '')).casefold()
        side = 1 if 'barcelona' in team else (-1 if 'athletic' in team else 0)
        a, b = max(0, e['start_frame']), min(n, e['end_frame'])
        j = codes.index(code)
        presence[a:b, j] += 1
        if code in positive or code in negative:
            votes[a:b, j] += side * (-1 if code in negative else 1)
    return votes, presence, codes, counts

def main():
    f = load(raw=True); g = f['half'] > 0; n = len(g)
    v, present, codes, counts = tag_features(n)
    ok, y = pseudo_labels(f)
    bp = f['ball_present'] > 0
    starts = np.flatnonzero(np.diff(np.r_[False, ~bp & g, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, ~bp & g, False].astype(int)) == -1)
    lengths = (ends - starts) / FPS
    tag = v.sum(1)
    report = {'graded_frames': int(g.sum()), 'missing_ball_fraction': float((~bp)[g].mean()),
        'gap_count': len(starts), 'median_gap_seconds': float(np.median(lengths)),
        'missing_time_in_gaps_over_5s': float(lengths[lengths > 5].sum() / lengths.sum()),
        'pseudo_label_frames': int(ok.sum()), 'clear_touch_clusters': len(clusters(f)),
        'tag_counts': counts, 'tags': {}}
    for j, code in enumerate(codes):
        covered = v[:, j] != 0
        test = ok & covered
        report['tags'][code] = {'coverage': float(covered[g].mean()), 'pseudo_n': int(test.sum()),
            'touch_agreement': float(((v[test, j] > 0) == y[test]).mean()) if test.any() else None}
    cov = tag != 0; test = ok & cov
    report['combined_tags'] = {'coverage': float(cov[g].mean()),
        'coverage_missing_ball': float(cov[g & ~bp].mean()),
        'touch_agreement': float(((tag[test] > 0) == y[test]).mean()),
        'conflicting_coverage': float(((v > 0).any(1) & (v < 0).any(1))[g].mean())}
    report['half_profiles'] = {str(h): {'missing_ball': float((~bp)[f['half'] == h].mean()),
        'median_players': float(np.median(f['n_players'][f['half'] == h]))} for h in (1, 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'eda.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
