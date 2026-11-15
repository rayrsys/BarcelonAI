"""Small external-only timing experiment; never overwrites the current submission."""
import json
import numpy as np
from cook import improve
from cook.model import decode,score,swap_features
from cook.external import prepare, ROOT
OUT=ROOT/'cook/reports/v3'
ART=ROOT/'data/cook/v3'

def native(model,f,symmetric=False):
    X,_=improve.design_v2(f)
    p=model.predict_proba(X)
    if symmetric:
        q=model.predict_proba(improve.design_v2(swap_features(f))[0])[:,[1,0,2]]
        p=(p+q)/2
    return p

def main():
    cfg=json.loads((improve.OUT/'frozen_config.json').read_text())
    model=improve.trained_model(cfg);T=np.array(cfg['T'])
    f,y=prepare(improve.DEV);p=native(model,f)
    q=model.predict_proba(improve.design_v2(swap_features(f))[0])[:,[1,0,2]]
    rows=[]
    for stride in (5,1):
        for sym in (False,True):
            prob=(p+q)/2 if sym else p
            if stride==5:
                ix=np.arange(0,len(p),5)
                prob=np.column_stack([np.interp(np.arange(len(p)),ix,prob[ix,j]) for j in range(3)])
            for evidence in (1.,1.5,2.):
                lab=decode(prob,f['half'],T,evidence)
                row={'stride':stride,'symmetric':sym,'evidence':evidence,**score(y,lab)}
                rows.append(row);print(json.dumps(row),flush=True)
    best=max(rows,key=lambda r:(r['spell_f1_mean'],r['macro_f1']))
    result={'development_match':improve.DEV,'candidates':rows,'selected':best,
            'baseline':rows[0],'development_gain':best['spell_f1_mean']-rows[0]['spell_f1_mean'],
            'competition_gt_read':False,'submission_changed':False}
    (OUT/'timing_development.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Selected:',json.dumps(best),flush=True)
if __name__=='__main__':main()
