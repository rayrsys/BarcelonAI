"""Download the openly licensed IDSSE data through Figshare's public API."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

FILES = {
    'J03WPY': {'meta': 51643487, 'tracking': 51643526},
    'J03WN1': {'meta': 51643472, 'tracking': 51643517},
    'J03WMX': {'meta': 51643475, 'tracking': 51643514},
    'J03WOH': {'meta': 51643478, 'tracking': 51643520},
    'J03WQQ': {'meta': 51643484, 'tracking': 51643529},
    'J03WOY': {'meta': 51643481, 'tracking': 51643523},
    'J03WR9': {'meta': 51643490, 'tracking': 51643532},
}
OUT = Path(__file__).resolve().parents[1] / 'data' / 'dfl'

def download(item):
    match, kind, fid = item
    path = OUT / f'{match}_{kind}.xml'
    url = f'https://api.figshare.com/v2/file/download/{fid}'
    if not path.exists():
        print(f'Downloading {match} {kind}', flush=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'MTAH-research/1.0'})
        temp = path.with_suffix('.part')
        with urllib.request.urlopen(req, timeout=90) as r, temp.open('wb') as out:
            head = r.read(1024)
            if b'<?xml' not in head:
                raise ValueError(f'{url} did not return XML')
            out.write(head)
            while chunk := r.read(1024 * 1024):
                out.write(chunk)
        temp.replace(path)
    sha = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    print(f'Ready {path.name}: {path.stat().st_size / 1e6:.1f} MB', flush=True)
    return {'file': path.name, 'url': url, 'sha256': sha.hexdigest(), 'bytes': path.stat().st_size}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('matches', nargs='*', default=list(FILES))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(m, k, f) for m in args.matches for k, f in FILES[m].items()]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, tasks))
    (OUT / 'downloads.json').write_text(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
