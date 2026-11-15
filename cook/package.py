"""Create an allowlisted public source package. Never walk/copy the whole data folder."""
from pathlib import Path
import hashlib
import json
import zipfile
from cook.train import ROOT, OUT

def main():
    paths = []
    for folder in ('cook','mtah','starter','evaluation'):
        paths.extend((ROOT/folder).glob('*.py'))
    paths += [ROOT/'cook/viewer_template.html',ROOT/'mtah/viewer_template.html']
    paths += [p for p in (ROOT/'cook/reports').rglob('*.json') if p.name != 'package_manifest.json']
    paths += list((ROOT/'cook/reports').rglob('protocol.md'))
    paths += list((ROOT/'models').glob('*'))
    paths.append(ROOT/'output/pdf/BarcelonAI.pdf')
    for name in ('games.py','requirements-cook.txt','requirements.txt','requirements-mtah.txt',
                 'COOK_README.md','README.md','TECHNICAL_DOCUMENT.md','BarcelonAI.md',
                 'SUBMISSION_READY.md','DATA.md','RULES.md','EVALUATION.md','SUBMISSION.md',
                 'CHALLENGE.md','FAQ.md','LICENSE','CITATION.cff','.gitignore','data/README.md'):
        paths.append(ROOT/name)
    # Published organiser assets only, never competition input or our generated frame arrays.
    paths += list((ROOT/'evaluation/gt').glob('*_public*'))
    paths += list((ROOT/'evaluation/results').glob('baseline_*.csv'))
    archive = ROOT/'deliverables/BarcelonAI_source.zip'; archive.parent.mkdir(exist_ok=True)
    entries = []
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)):
            if not p.is_file(): continue
            rel = str(p.relative_to(ROOT))
            assert not rel.startswith('data/') or rel == 'data/README.md'
            assert p.suffix not in ('.mp4','.npz','.npy','.xml')
            name = 'README.md' if rel == 'COOK_README.md' else ('ORGANISERS_README.md' if rel == 'README.md' else rel)
            z.write(p,name)
            entries.append({'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    report = {'archive':str(archive.relative_to(ROOT)), 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
              'file_count':len(entries),'bytes':archive.stat().st_size,'files':entries,
              'competition_raw_data_included':False,'generated_viewer_included':False}
    (OUT/'package_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))

if __name__ == '__main__':
    main()
