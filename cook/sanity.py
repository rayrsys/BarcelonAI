"""Post-freeze public-window alignment check. Never import this from training.

The result must not be used to revise/select the submission. It is not an official ranking.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluation import possession_metrics as pm
from cook.train import ROOT, OUT, ART

def main():
    config_path = OUT/'frozen_config.json'
    manifest = json.loads((OUT/'submission_manifest.json').read_text())
    candidate = ROOT / manifest['submission']
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == manifest['sha256']
    gt = pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    result = {'purpose':'alignment/format check only; not used for model selection',
              'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(), 'scores':{}}
    for name in ('baseline_smooth50','baseline_smooth125','v9','cook_v1'):
        df = pm.read_label_csv(ROOT / f'evaluation/results/{name}.csv')
        reason = pm.validate_submission(df,199319)
        if reason is not None:
            raise ValueError(f'{name}: {reason}')
        r = pm.evaluate(gt,pm.label_codes(df),bootstrap_n=1000)
        result['scores'][name] = {k:r[k] for k in ('macro_f1','spell_f1_mean','spell_bootstrap_low','spell_bootstrap_high','n_pred_spells')}
    with np.load(ART/'target_predictions.npz') as z:
        old = pm.read_label_csv(ROOT/'evaluation/results/v9.csv')['label'].values
        result['regenerated_reference_agreement_with_saved_v9'] = float((z['reference'] == old).mean())
    (OUT/'public_sanity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
