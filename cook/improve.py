"""Version 2 experiment. All tuning remains on external matches; v1 stays frozen.

Train on four previously used external games, tune on one new game, then test on two
previously untouched games. Never reads competition ground truth or the public score file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from cook.external import prepare, ROOT
from cook.model import design, shift, swap_features, make_model, transitions, decode, score
from cook.train import reference, blend, model_file

ART=ROOT/'data/cook/v2'; OUT=ROOT/'cook/reports/v2'
TRAIN=('J03WPY','J03WN1','J03WMX','J03WOH')
DEV='J03WQQ'; TEST=('J03WOY','J03WR9')

def trained_model(cfg):
    path=ART/f"{cfg['selected']['model']}.joblib"
    if not path.exists():path=ROOT/'models'/f"v2_{cfg['selected']['model']}.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['model_sha256']:
        raise ValueError('Model differs from the frozen v2 artifact')
    return joblib.load(path)

def design_v2(f):
    X,names=design(f);cols=[X]
    for k in ('d_home','d_away','ball_llr_home','ball_present','ball_speed'):
        for offset in (-12,-5,5,12):
            cols.append(shift(f[k],offset,f['half'])[:,None]);names.append(f'{k}_short{offset}')
    out=np.column_stack(cols).astype(np.float32);out[~np.isfinite(out)]=np.nan
    return out,names

def probabilities(model,f):
    X,_=design_v2(f);ix=np.arange(0,len(X),5);p=model.predict_proba(X[ix])
    return np.column_stack([np.interp(np.arange(len(X)),ix,p[:,j]) for j in range(3)])

def spell_weight(y,h):
    """Give shorter training spells more influence. Computed from training targets only."""
    starts=np.r_[0,np.flatnonzero((y[1:]!=y[:-1]) | (h[1:]!=h[:-1]))+1]
    ends=np.r_[starts[1:],len(y)];weight=np.ones(len(y))
    for a,b in zip(starts,ends):
        if y[a]>=0:weight[a:b]=np.clip(np.sqrt(12/((b-a)/25)),.5,3)
    return weight

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=float)+'\n')

def frozen_v1(f,q):
    from cook.model import probabilities as p1
    cfg=json.loads((ROOT/'cook/reports/frozen_config.json').read_text())
    s=cfg['selected'];m=joblib.load(model_file(s['leaves']))
    return decode(blend(p1(m,f),q,s['weight']),f['half'],np.array(cfg['T']),s['evidence'])

def train():
    xs,ys,ws,targets,halves=[],[],[],[],[]
    for match in TRAIN:
        for seed in (2027,2028):
            print('Prepare',match,seed,flush=True)
            f,y=prepare(match,seed);X,names=design_v2(f)
            take=(y>=0)&(np.arange(len(y))%5==0);w=spell_weight(y,f['half'])[take]
            xs.extend((X[take],design_v2(swap_features(f))[0][take]))
            ys.extend((y[take],np.where(y[take]<2,1-y[take],2)));ws.extend((w,w))
            if seed==2027:targets.append(y);halves.append(f['half'])
    X,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)
    T=transitions(targets,halves);fdev,ydev=prepare(DEV);oldlab,qdev=reference(DEV,fdev)
    baseline=score(ydev,frozen_v1(fdev,qdev));print('Frozen v1 development:',baseline,flush=True)
    rows=[];best=None
    for name,leaves,weighted in (('leaf15',15,False),('leaf31',31,False),('short_spells31',31,True)):
        print('Training',name,X.shape,flush=True)
        model=make_model(leaves).set_params(max_iter=300)
        model.fit(X,y,sample_weight=w if weighted else None)
        joblib.dump(model,ART/f'{name}.joblib');p=probabilities(model,fdev)
        for weight in (.5,.75,1.):
            for evidence in (.5,1.,2.,4.):
                lab=decode(blend(p,qdev,weight),fdev['half'],T,evidence)
                r={'model':name,'weight':weight,'evidence':evidence,**score(ydev,lab)};rows.append(r)
                if best is None or (r['spell_f1_mean'],r['macro_f1'])>(best['spell_f1_mean'],best['macro_f1']):best=r
        save('development.json',{'v1':baseline,'candidates':rows,'selected':best})
        print('Best development:',best,flush=True)
    cfg={'train_matches':TRAIN,'development_match':DEV,'test_matches':TEST,'training_rows':len(y),
         'feature_names':names,'T':T.tolist(),'selected':best,'no_competition_gt_read':True,
         'model_sha256':hashlib.sha256((ART/f"{best['model']}.joblib").read_bytes()).hexdigest()}
    save('frozen_config.json',cfg)
    print('V2 frozen; both new test matches remain unscored.',flush=True)

def test():
    cfg=json.loads((OUT/'frozen_config.json').read_text());s=cfg['selected']
    model=trained_model(cfg);results=[]
    for match in TEST:
        f,y=prepare(match);oldlab,q=reference(match,f)
        v1=frozen_v1(f,q);v2=decode(blend(probabilities(model,f),q,s['weight']),f['half'],np.array(cfg['T']),s['evidence'])
        r={'match':match,'v1':score(y,v1,300),'v2':score(y,v2,300),'reference_v9':score(y,oldlab,300)}
        results.append(r);print(json.dumps(r,indent=2),flush=True)
        np.savez_compressed(ART/f'{match}_test.npz',v1=v1,v2=v2,target=y)
    avg={k:float(np.mean([r[k]['spell_f1_mean'] for r in results])) for k in ('v1','v2','reference_v9')}
    save('external_test.json',{'matches':results,'mean_spellF1':avg,
         'v2_wins_both':all(r['v2']['spell_f1_mean']>r['v1']['spell_f1_mean'] for r in results)})
    print('Two-match mean:',avg,flush=True)

def predict():
    from mtah.features import load
    from mtah import pipeline
    cfg=json.loads((OUT/'frozen_config.json').read_text());s=cfg['selected']
    model=trained_model(cfg);f=load(raw=True)
    if s['weight']==1.:
        q=np.full((len(f['half']),3),1/3)
    else:
        _,_,q=pipeline.run(dict(pipeline.DEFAULTS),verbose=False,return_post=True)
    labels=decode(blend(probabilities(model,f),q,s['weight']),f['half'],np.array(cfg['T']),s['evidence'])
    path=Path(pipeline.write(labels,'BarcelonAI_v2'))
    save('submission_manifest.json',{'submission':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'model_sha256':cfg['model_sha256'],'competition_public_evaluation_performed':False})
    print('V2 candidate generated without reading competition GT:',path,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('train','test','predict'),default='train');args=ap.parse_args()
    ART.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    {'train':train,'test':test,'predict':predict}[args.stage]()
