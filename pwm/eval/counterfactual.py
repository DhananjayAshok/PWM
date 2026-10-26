"""Counterfactual accuracy. A test group is one start state with the strong policy at every counterfactual noise level
(5 rollouts, identical first frame). For every ordered pair (A, B) of rollouts in a group: is the prediction made from
A's commands closer to A's real future than to B's real future? Distances are in the model's own feature space (each
model's features are compared only with its own features, so the score is a ranking and scale-free across models).
"""
from itertools import permutations

import numpy as np

from pwm.eval.probe import flatten_features


def feature_distance(a, b):
    """Mean L2 distance between two feature sequences of equal length (per-step features flattened and pooled)."""
    a, b = flatten_features(a).numpy(), flatten_features(b).numpy()
    n = min(len(a), len(b))
    return float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))


def group_pairs(*, preds, reals):
    """
    preds, reals: {episode_id: feature sequence} for one group (aligned on the same frame indices).
    Returns a list of (A, B, correct) for every ordered pair.
    """
    out = []
    for a, b in permutations(sorted(preds), 2):
        if a not in reals or b not in reals:
            continue
        out.append((a, b, feature_distance(preds[a], reals[a]) < feature_distance(preds[a], reals[b])))
    return out
