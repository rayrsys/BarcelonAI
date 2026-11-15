"""Build a local viewer: the broadcast video with the model's call overlaid in sync.

    .venv/bin/python -m mtah.viewer            # writes data/viewer/index.html (gitignored, local only)

Labels live on the tactical timeline; the broadcast-to-tactical offsets from mtah.broadcast map them
onto the broadcast video. Stretches the alignment could not verify (replays, close-ups) show "no sync".
"""
import json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import broadcast as B  # noqa: E402
from games import source_frames  # noqa: E402


def on_broadcast(labels, offs, nb):
    out = np.full(nb, "?", dtype="<U1")
    for a, o in offs.items():
        t = np.arange(a, a + B.CHUNK) + o; ok = (t >= 0) & (t < len(labels))
        out[a:a + B.CHUNK][ok] = labels[t[ok]]
    out[out == "X"] = "?"
    return out


def rle(x):
    ch = np.concatenate(([0], np.where(x[1:] != x[:-1])[0] + 1))
    return [[int(i), str(x[i])] for i in ch]


def main(model="v8", base="baseline_smooth50"):
    nb = source_frames("broadcast"); offs = B.offsets()
    rd = lambda n: pd.read_csv(os.path.join(ROOT, "evaluation", "results", n + ".csv"))["label"].values.astype(str)
    data = {"n": nb, "fps": 25, "model": rle(on_broadcast(rd(model), offs, nb)), "base": rle(on_broadcast(rd(base), offs, nb)),
            "modelName": model}
    html = open(os.path.join(ROOT, "mtah", "viewer_template.html")).read().replace("/*DATA*/", json.dumps(data))
    out = os.path.join(ROOT, "data", "viewer"); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "index.html"), "w").write(html)
    synced = float(np.mean(on_broadcast(rd(model), offs, nb) != "?"))
    print("wrote data/viewer/index.html  synced share of broadcast video: %.2f" % synced)


if __name__ == "__main__":
    main(*sys.argv[1:])
