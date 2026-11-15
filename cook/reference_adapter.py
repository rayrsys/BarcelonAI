"""Run the existing v9 algorithm on external matches without changing its source.

The only unavailable external asset is broadcast tracking. Its fallback is therefore disabled.
All geometric heuristics, EM and self-training remain enabled. No external targets are read.
"""
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
from mtah import pipeline, shape_model, kickoff, restarts

def run(d, f):
    X, _ = shape_model.design(f)
    p = shape_model.blocked_cv(f, X)
    cfg = dict(pipeline.DEFAULTS, ball_fill=0)
    with ExitStack() as stack:
        stack.enter_context(patch.object(pipeline, 'load_features', return_value=f))
        stack.enter_context(patch.object(pipeline, 'load_oof', return_value=p))
        for mod in (kickoff, restarts):
            stack.enter_context(patch.object(mod, 'load_atd', return_value=d))
            stack.enter_context(patch.object(mod, 'PITCH', d['pitch']))
        lab, info, post = pipeline.run(cfg, verbose=False, return_post=True)
    return lab, post, info
