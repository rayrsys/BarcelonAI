"""Train -> select on an external development match -> freeze -> external test.

No competition ground truth is read. The public sanity check is a separate explicit command.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
from cook.external import prepare, parse, corrupt, ROOT, DATA
from cook.model import design, make_model, probabilities, transitions, decode, score, swap_features

OUT = ROOT / 'cook' / 'reports'
ART = ROOT / 'data' / 'cook'
TRAIN = ('J03WPY', 'J03WN1')
DEV = 'J03WMX'
TEST = 'J03WOH'

def model_file(leaves):
    local = ART / f'boost_{leaves}.joblib'
    return local if local.exists() else ROOT / 'models' / f'cook_boost_{leaves}.joblib'

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: float(x)) + '\n')

def reference(match, f):
    path = ART / f'{match}_reference.npz'
    if path.exists():
        with np.load(path) as z:
            return z['labels'], z['post']
    from cook.reference_adapter import run
    print('Running Previous model v9 external adapter:', match, flush=True)
    lab, post, info = run(corrupt(parse(match)), f)
    np.savez_compressed(path, labels=lab, post=post)
    return lab, post

def blend(p, q, weight):
    q = np.nan_to_num(q, nan=1/3)
    q = q / np.maximum(q.sum(1, keepdims=True), 1e-9)
    return weight * p + (1-weight) * q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('train','test','predict'), default='train')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    if args.stage == 'predict':
        return predict()
    if args.stage == 'test':
        return test()
    xs, ys, targets, halves = [], [], [], []
    for m in TRAIN:
        # Two sensor corruptions per training game, fixed independently of validation scores.
        for seed in (2027, 2028):
            print('Preparing training:', m, seed, flush=True)
            f, y = prepare(m, seed); X, names = design(f)
            take = (y >= 0) & (np.arange(len(y)) % 5 == 0)
            xs.append(X[take]); ys.append(y[take])
            xs.append(design(swap_features(f))[0][take])
            ys.append(np.where(y[take] < 2, 1-y[take], 2))
            if seed == 2027:
                targets.append(y); halves.append(f['half'])
    Xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    T = transitions(targets, halves)
    print('Training rows/features:', Xtr.shape, flush=True)
    fdev, ydev = prepare(DEV)
    clab, qdev = reference(DEV, fdev)
    baseline = score(ydev, clab)
    print('Reference external dev:', baseline, flush=True)
    results = []; best = None
    for leaves in (15, 31):
        print('Training boosted trees, leaves:', leaves, flush=True)
        model = make_model(leaves).fit(Xtr, ytr)
        p = probabilities(model, fdev)
        joblib.dump(model, ART / f'boost_{leaves}.joblib')
        for weight in (.5, .75, 1.0):
            for evidence in (.1, .25, .5, 1.0):
                labels = decode(blend(p,qdev,weight), fdev['half'], T, evidence)
                result = {'leaves':leaves, 'weight':weight, 'evidence':evidence, **score(ydev, labels)}
                results.append(result)
                if best is None or (result['spell_f1_mean'],result['macro_f1']) > (best['spell_f1_mean'],best['macro_f1']):
                    best = result
        print('Current best external dev:', best, flush=True)
        save_json(OUT/'development.json', {'reference_v9': baseline, 'candidates': results, 'selected': best})
    manifest = {'train_matches': list(TRAIN), 'development_match':DEV, 'test_match':TEST,
                'selected':best, 'T':T.tolist(), 'feature_names':names,
                'sensor_seeds':[2027,2028], 'training_rows':len(ytr),
                'team_swap_augmentation':True,
                'no_competition_gt_read':True, 'selection_metric':'external development spellF1'}
    model_path = ART / f"boost_{best['leaves']}.joblib"
    manifest['model_sha256'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    save_json(OUT/'frozen_config.json', manifest)
    print('Frozen config written. Test match has not been scored.', flush=True)

def test():
    config = json.loads((OUT/'frozen_config.json').read_text()); best = config['selected']
    model = joblib.load(model_file(best['leaves']))
    f, y = prepare(TEST)
    clab, q = reference(TEST, f)
    p = probabilities(model, f)
    labels = decode(blend(p,q,best['weight']),f['half'],np.array(config['T']),best['evidence'])
    r = {'match':TEST, 'selected_before_test':best, 'reference_v9':score(y,clab,300),
         'cook':score(y,labels,300)}
    # Organisers' nearest-player baseline, on exactly the same degraded input.
    from starter.baseline import smooth
    near = np.where(np.isnan(f['d_home']) | np.isnan(f['d_away']),-1,
                    np.where(f['d_home'] < f['d_away'],0,1)).astype(np.int8)
    for w in (50,125):
        lab = smooth(near,w); r[f'baseline_smooth{w}'] = score(y, lab, 300)
    save_json(OUT/'external_test.json',r)
    np.savez_compressed(ART/'external_test_predictions.npz', labels=labels, reference=clab, target=y)
    print(json.dumps(r,indent=2),flush=True)

def predict():
    from mtah.features import load
    from mtah import pipeline
    config = json.loads((OUT/'frozen_config.json').read_text()); best = config['selected']
    model_path = model_file(best['leaves'])
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != config['model_sha256']:
        raise ValueError('Model differs from the frozen artifact')
    model = joblib.load(model_path)
    f = load(raw=True)
    p = probabilities(model,f)
    # Generate the baseline from code; never treat a historical prediction CSV as labels.
    clab, _, q = pipeline.run(dict(pipeline.DEFAULTS),verbose=False,return_post=True)
    labels = decode(blend(p,q,best['weight']),f['half'],np.array(config['T']),best['evidence'])
    path = Path(pipeline.write(labels,'cook_v1'))
    team_path = path.with_name('BarcelonAI.csv')
    shutil.copyfile(path,team_path)
    np.savez_compressed(ART/'target_predictions.npz',labels=labels,reference=clab,probability=p)
    from evaluation import possession_metrics as pm
    codes = pm.label_codes(labels); runs = pm.valid_runs(codes)
    stats = pm.possession_stats(codes,runs)
    manifest = {'team':'BarcelonAI', 'submission':str(team_path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'model_sha256':config['model_sha256'], 'possessions':stats['n_possessions'] if 'n_possessions' in stats else len(pm.possession_spells(codes,runs)),
                'label_counts':{c:int((labels==c).sum()) for c in 'HADX'}}
    save_json(OUT/'submission_manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)

if __name__ == '__main__':
    main()
