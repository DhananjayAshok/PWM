"""Readers for episodes and predictions."""
import json
import os

import imageio.v3 as iio
import numpy as np


def read_video(path):
    return np.asarray(iio.imread(path, plugin="pyav"))           # (T, H, W, 3) uint8


def episode_rows(*, data_dir, robot, dataset, split):
    path = os.path.join(data_dir, robot, dataset, split, "episodes.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def episode_paths(*, data_dir, robot, dataset, split, episode_id):
    base = os.path.join(data_dir, robot, dataset, split, episode_id)
    return base + ".mp4", base + ".npz"


def prediction_dir(*, storage_dir, model, robot, dataset, tag, episode_id=None):
    d = os.path.join(storage_dir, "predictions", model, robot, dataset, tag)
    return d if episode_id is None else os.path.join(d, episode_id)


def real_features_path(*, storage_dir, model, robot, tag, dataset, split, episode_id):
    return os.path.join(storage_dir, "predictions", model, robot, "features", tag, dataset, split, episode_id + ".npz")


def load_prediction(path):
    out = {"pred_features": np.load(os.path.join(path, "pred_features.npy")),
           "frame_index": np.load(os.path.join(path, "frame_index.npy"))}
    with open(os.path.join(path, "meta.json")) as f:
        out["meta"] = json.load(f)
    video = os.path.join(path, "pred.mp4")
    out["video"] = video if os.path.exists(video) else None
    return out
