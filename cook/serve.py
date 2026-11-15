"""Private loopback-only dashboard server with byte ranges for the large local video."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit, unquote
import re

ROOT = Path(__file__).resolve().parents[1]

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):
        super().__init__(*a,directory=str(ROOT),**kw)

    def log_message(self,*args):
        pass

    def send_head(self):
        requested = unquote(urlsplit(self.path).path)
        if requested == '/':
            self.send_response(302); self.send_header('Location','/data/viewer/index.html'); self.end_headers(); return None
        path = Path(self.translate_path(self.path)).resolve()
        allowed = (path.parent == ROOT/'data/viewer' or path.parent == ROOT/'cook/reports' or
                   path == ROOT/'evaluation/results/cook_v1.csv' or
                   path == ROOT/'evaluation/results/BarcelonAI.csv' or
                   path == ROOT/'deliverables/BarcelonAI_source.zip' or
                   path == ROOT/'data/broadcast/BAR-ATH_broadcast.mp4' or
                   path == ROOT/'SUBMISSION_READY.md' or path == ROOT/'TECHNICAL_DOCUMENT.md')
        allowed = allowed or path == ROOT/'BarcelonAI.md'
        if not allowed or not path.is_file():
            self.send_error(404); return None
        size = path.stat().st_size
        self._range = None
        header = self.headers.get('Range')
        if header:
            m = re.fullmatch(r'bytes=(\d*)-(\d*)',header)
            if not m or (not m[1] and not m[2]):
                self.send_error(416); return None
            if m[1]:
                start = int(m[1]); end = int(m[2]) if m[2] else size-1
            else:
                start = max(0,size-int(m[2])); end = size-1
            end = min(end,size-1)
            if start > end or start >= size:
                self.send_error(416); return None
            self.send_response(206)
            self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self._range = (start,end)
        else:
            start,end = 0,size-1; self.send_response(200)
        self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('Cache-Control','no-cache')
        self.end_headers()
        f = path.open('rb'); f.seek(start); return f

    def copyfile(self,source,output):
        remaining = self._range[1]-self._range[0]+1 if self._range else None
        try:
            while remaining is None or remaining > 0:
                chunk = source.read(min(1024*1024,remaining) if remaining is not None else 1024*1024)
                if not chunk: break
                output.write(chunk)
                if remaining is not None: remaining -= len(chunk)
        except (BrokenPipeError,ConnectionResetError):
            pass

if __name__ == '__main__':
    print('Viewer: http://127.0.0.1:8765/data/viewer/index.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
