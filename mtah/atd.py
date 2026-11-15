"""Parse the raw FIFA/EPTS ATD into dense numpy arrays and cache them.

Arrays are indexed by VIDEO frame (ATD frame_id - 1), so row i lines up with submission row i.
Coordinates stay normalised 0-1; metres = coord * PITCH.
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache")
PITCH = np.array(GAMES[GAME]["pitch"], dtype=np.float32)  # (105.3, 68.0)
HOME, AWAY = 0, 1


def team_of_slot(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def parse(source="tactical"):
    paths = asset_paths(source=source)
    n = source_frames(source)
    team = team_of_slot(paths["atd_meta"])
    k = len(team)
    players = np.full((n, k, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    with open(paths["atd_raw"]) as fh:
        for line in fh:
            fid, pl, bl = line.rstrip("\n").split(":")
            i = int(fid) - 1
            players[i] = np.array(pl.replace(";", ",").split(","), dtype=np.float32).reshape(k, 3)
            ball[i] = np.array(bl.split(","), dtype=np.float32)
    return {"players": players, "ball": ball, "team": team}


def ball_mode():
    """'' = provided tactical ball only; 'fill' = gaps filled from the aligned broadcast tracking."""
    return os.environ.get("MTAH_BALL", "")


def load(source="tactical", raw=None):
    path = os.path.join(CACHE, source + "_atd.npz")
    if not os.path.isfile(path):
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, **parse(source))
    z = np.load(path)
    d = {key: z[key] for key in z.files}
    if raw is None:
        raw = ball_mode() != "fill"
    if source == "tactical" and not raw:
        from mtah.broadcast import fill
        fb = fill(); miss = ~np.isfinite(d["ball"][:, 0]) & np.isfinite(fb[:, 0])
        d["ball"] = d["ball"].copy(); d["ball"][miss, :2] = fb[miss]
    return d


if __name__ == "__main__":
    for src in sys.argv[1:] or ["tactical"]:
        d = load(src)
        print(src, {k: v.shape for k, v in d.items()})
