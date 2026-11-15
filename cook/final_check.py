"""One post-selection public sanity check for the promoted submission. Not training."""
import hashlib
import json
from cook.train import ROOT
from evaluation import possession_metrics as pm

def main():
    release_path=ROOT/'cook/reports/release.json'
    release=json.loads(release_path.read_text())
    path=ROOT/release['submission']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==release['sha256']
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    results={}
    for name in ('baseline_raw','baseline_smooth25','baseline_smooth50','baseline_smooth125','v9','cook_v1','BarcelonAI'):
        df=pm.read_label_csv(ROOT/f'evaluation/results/{name}.csv')
        error=pm.validate_submission(df,199319)
        if error:raise ValueError(f'{name}: {error}')
        r=pm.evaluate(gt,pm.label_codes(df),bootstrap_n=1000)
        results[name]={k:r[k] for k in ('spell_f1_mean','macro_f1','spell_bootstrap_low','spell_bootstrap_high','n_pred_spells')}
    out={'purpose':'pipeline sanity check after external-only selection; not used to change or select models',
         'release_sha256_before_check':hashlib.sha256(release_path.read_bytes()).hexdigest(),'scores':results}
    (ROOT/'cook/reports/v2/public_sanity.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
