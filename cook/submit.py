"""Regenerate the externally selected team submission. Never reads competition labels."""
import hashlib
import json
import shutil
from pathlib import Path
from cook.train import ROOT
from cook import improve

def main():
    report=json.loads((improve.OUT/'external_test.json').read_text())
    if not report['v2_wins_both']:
        raise RuntimeError('The v2 candidate did not pass the frozen promotion rule')
    improve.ART.mkdir(parents=True,exist_ok=True)
    improve.predict()
    source=ROOT/'evaluation/results/BarcelonAI_v2.csv'
    target=source.with_name('BarcelonAI.csv');shutil.copyfile(source,target)
    cfg=json.loads((improve.OUT/'frozen_config.json').read_text())
    release={'team':'BarcelonAI','version':2,'submission':str(target.relative_to(ROOT)),
             'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
             'model_sha256':cfg['model_sha256'],'selection':'higher spellF1 than frozen v1 on both untouched external tests',
             'mean_external_spellF1':report['mean_spellF1']}
    (ROOT/'cook/reports/release.json').write_text(json.dumps(release,indent=2)+'\n')
    print(json.dumps(release,indent=2))

if __name__=='__main__':main()
