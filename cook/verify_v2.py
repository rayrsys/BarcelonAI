"""Verify v2 from a new raw-data cache, with the public pretrained model artifact."""
from contextlib import ExitStack
from unittest.mock import patch
import hashlib
import json
import numpy as np
from mtah import atd,features
from cook import improve
from cook.train import ROOT
from cook.model import decode

def main():
    cfg=json.loads((improve.OUT/'frozen_config.json').read_text())
    assert cfg['selected']['weight']==1., 'This verification is for the frozen standalone v2 model'
    cache=ROOT/'data/cook/v2/fresh_cache';cache.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for mod in (atd,features):stack.enter_context(patch.object(mod,'CACHE',str(cache)))
        # Force public-model fallback to verify the distributed model, not the training copy.
        stack.enter_context(patch.object(improve,'ART',cache))
        f=features.load(raw=True);p=improve.probabilities(improve.trained_model(cfg),f)
        labels=decode(p,f['half'],np.array(cfg['T']),cfg['selected']['evidence'])
    content='frame,label\n'+''.join(f'{i},{v}\n' for i,v in enumerate(labels))
    digest=hashlib.sha256(content.encode()).hexdigest()
    expected=json.loads((ROOT/'cook/reports/release.json').read_text())['sha256']
    result={'raw_asset_rebuild':True,'public_model_artifact_used':True,'sha256':digest,'matches_submission':digest==expected}
    (improve.OUT/'clean_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert digest==expected

if __name__=='__main__':main()
