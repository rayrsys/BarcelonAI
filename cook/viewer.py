"""Build the private local HTML dashboard; match data never leaves this machine."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from games import source_frames
from mtah import broadcast
from mtah.viewer import on_broadcast, rle
from cook.train import ROOT, OUT, ART

def main():
    dest = ROOT/'data/viewer'; dest.mkdir(parents=True,exist_ok=True)
    old = dest/'index.html'; backup = dest/'reference_original.html'
    if old.exists() and not backup.exists():
        shutil.copy2(old,backup)
    path = ART/'viewer_offsets.json'
    if path.exists():
        offs = {int(k):int(v) for k,v in json.loads(path.read_text()).items()}
    else:
        offs = broadcast.offsets(); path.write_text(json.dumps({int(k):int(v) for k,v in offs.items()}))
    n = source_frames('broadcast')
    arrays = {}
    for name in ('BarcelonAI','v9','baseline_smooth50'):
        labels = pd.read_csv(ROOT/f'evaluation/results/{name}.csv')['label'].to_numpy(dtype=str)
        arrays[name] = on_broadcast(labels,offs,n)
    valid = (arrays['BarcelonAI'] != '?') & (arrays['v9'] != '?')
    diff = valid & (arrays['BarcelonAI'] != arrays['v9'])
    starts = np.flatnonzero(np.diff(np.r_[False,diff,False].astype(int))==1)
    ends = np.flatnonzero(np.diff(np.r_[False,diff,False].astype(int))==-1)
    starts = starts[(ends-starts)>=25]
    first = np.flatnonzero(np.isin(arrays['BarcelonAI'],['H','A']))
    tested=json.loads((OUT/'v2/external_test.json').read_text())
    baselines=json.loads((OUT/'v2/external_baselines.json').read_text())
    external={
        'cook':{'spell_f1_mean':tested['mean_spellF1']['v2']},
        'reference_v9':{'spell_f1_mean':tested['mean_spellF1']['reference_v9']},
        'baseline_smooth50':{'spell_f1_mean':float(np.mean([r['baseline_smooth50']['spell_f1_mean'] for r in baselines.values()]))}}
    data = {'n':n,'fps':25,'models':{k:rle(v) for k,v in arrays.items()},
            'start':int(first[0]) if len(first) else 0,'disagreements':starts.tolist(),
            'sync':float(valid.mean()),
            'external':external,
            'public':json.loads((OUT/'v2/public_sanity.json').read_text()),
            'manifest':json.loads((OUT/'release.json').read_text())}
    html = (ROOT/'cook/viewer_template.html').read_text().replace('/*DATA*/',json.dumps(data))
    old.write_text(html)
    print('Built local comparison viewer; synced fraction:',round(data['sync'],3))

if __name__ == '__main__':
    main()
