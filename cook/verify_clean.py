"""Rebuild inherited features and pseudo-models from raw match assets in an isolated cache."""
import hashlib
import json
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
import numpy as np
import joblib
from mtah import atd, features, shape_model, broadcast, pipeline
from cook.train import ROOT, OUT, ART, blend
from cook.model import probabilities, decode

def main():
    config = json.loads((OUT/'frozen_config.json').read_text()); best = config['selected']
    cache = ROOT/'data/cook/clean_cache'; cache.mkdir(parents=True,exist_ok=True)
    model = joblib.load(ART/f"boost_{best['leaves']}.joblib")
    with ExitStack() as stack:
        for module in (atd,features,shape_model,broadcast):
            stack.enter_context(patch.object(module,'CACHE',str(cache)))
        f = features.load(raw=True)
        p = probabilities(model,f)
        _,_,q = pipeline.run(dict(pipeline.DEFAULTS),verbose=False,return_post=True)
        labels = decode(blend(p,q,best['weight']),f['half'],np.array(config['T']),best['evidence'])
    original = np.load(ART/'target_predictions.npz')['labels']
    result = {'fresh_cache_frame_agreement':float((labels==original).mean()),
              'different_frames':int((labels!=original).sum()), 'raw_asset_rebuild':True}
    (OUT/'clean_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not np.array_equal(labels,original):
        raise AssertionError('Clean regeneration differs; do not claim raw-data reproducibility')

if __name__ == '__main__':
    main()
