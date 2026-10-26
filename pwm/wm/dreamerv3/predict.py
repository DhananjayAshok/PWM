"""Open-loop prediction with a trained DreamerV3 world model, in the plans/predictions_format.md layout.

``imagine``: observe the context frames (with the commands into them), then roll the RSSM forward on the remaining
commands alone; returns the imagined prior latent states (pred_features, the state probe's input) and decoded frames
at the model's resolution. ``encode_real``: posterior latent states of real frames (the encode entry, encode.py).
Upsampling to the evaluation resolution is left to the evaluation code.

Protocol (plans/predictions_format.md): window = the first 4.8 s from recorded frame 0 (the whole episode if
shorter), context = 1 real frame, free-running, seed 0; sample 0 is pred.mp4 / pred_features.npy, extra samples
pred_s<k>.*.

    python -m pwm.wm.dreamerv3.predict --model_dir <dir with best.pt> --robot franka_libero --dataset nominal \
        [--split test] [--checkpoint_tag <tag>] [--samples 1]
"""
import json
import os
import time
from typing import Any

import click
import imageio.v2 as iio
import numpy as np
import torch

from pwm.utils import load_parameters, log_info
from pwm.wm.dreamerv3.data import Replay, build_cache, read_index
from pwm.wm.dreamerv3.model import build_world_model, make_config, to_device


def load_model(*, model_dir: str, parameters: dict[str, Any]):
    with open(os.path.join(model_dir, "action_stats.json")) as f:
        stats = json.load(f)
    path = os.path.join(model_dir, "best.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    s = state["meta"]["settings"]
    config = make_config(settings=s, action_dim=state["meta"]["action_dim"])
    wm = build_world_model(config=config, image_size=s["image_size"])
    wm.load_state_dict(state["model"])
    wm.eval()
    return wm, stats, s


def _inputs(wm, frames: np.ndarray, actions: np.ndarray, stats: dict) -> dict[str, torch.Tensor]:
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    a = np.clip((actions - mean) / std, -10, 10).astype(np.float32)
    a[0] = 0.0
    is_first = np.zeros(len(frames), np.float32)
    is_first[0] = 1.0
    batch = {"image": frames[None], "action": a[None], "is_first": is_first[None],
             "is_terminal": np.zeros((1, len(frames)), np.float32)}
    return to_device(batch, wm._config.device)


@torch.no_grad()
def encode_real(*, wm, frames: np.ndarray, actions: np.ndarray, stats: dict) -> np.ndarray:
    """Posterior features (N, F) of real frames (N, H, W, 3) uint8 given the commands into them (N, A)."""
    data = _inputs(wm, frames, actions, stats)
    post, _ = wm.dynamics.observe(wm.encoder(data), data["action"], data["is_first"])
    return wm.dynamics.get_feat(post)[0].float().cpu().numpy()


@torch.no_grad()
def imagine(*, wm, context: np.ndarray, actions: np.ndarray, stats: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """``context`` (C, H, W, 3) uint8 real frames; ``actions`` (C + N, A): row i = commands into frame i.
    Returns ``feat`` (N, F) imagined latent features and ``frames`` (N, H, W, 3) uint8 for frames C .. C+N-1."""
    torch.manual_seed(seed)
    c = len(context)
    pad = np.zeros((len(actions) - c,) + context.shape[1:], np.uint8)  # images after the context are never read
    data = _inputs(wm, np.concatenate([context, pad]), actions, stats)
    post, _ = wm.dynamics.observe(wm.encoder({k: v[:, :c] for k, v in data.items()}), data["action"][:, :c],
                                  data["is_first"][:, :c])
    init = {k: v[:, -1] for k, v in post.items()}
    prior = wm.dynamics.imagine_with_action(data["action"][:, c:], init)
    feat = wm.dynamics.get_feat(prior)
    image = wm.heads["decoder"](feat)["image"].mode()
    frames = (image[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return {"feat": feat[0].float().cpu().numpy(), "frames": frames}


def window_frames(*, fps: float, horizon_s: float, n_frames: int, context: int) -> int:
    """Predicted frames for one episode: round(horizon_s * fps), or what the episode has after the context."""
    return max(0, min(int(round(horizon_s * fps)), n_frames - context))


def checkpoint_tag_of(model_dir: str) -> str:
    return os.path.basename(os.path.normpath(model_dir))


@click.command()
@click.option("--model_dir", required=True, help="Directory with best.pt and action_stats.json")
@click.option("--robot", required=True)
@click.option("--dataset", default="nominal")
@click.option("--split", default="test")
@click.option("--checkpoint_tag", default=None, help="Default: the model directory's name")
@click.option("--horizon_s", default=4.8, type=float)
@click.option("--context", default=1, type=int, help="Real context frames")
@click.option("--samples", default=1, type=int)
@click.option("--seed", default=0, type=int)
def main(model_dir, robot, dataset, split, checkpoint_tag, horizon_s, context, samples, seed):
    parameters = load_parameters()
    wm, stats, s = load_model(model_dir=model_dir, parameters=parameters)
    tag = checkpoint_tag or checkpoint_tag_of(model_dir)
    dataset_dir = os.path.join(parameters["data_dir"], robot, dataset)
    cache = build_cache(dataset_dir=dataset_dir, split=split, size=s["image_size"],
                        workers=int(os.environ.get("OMP_NUM_THREADS", "8")),
                        log=lambda m: log_info(m, parameters=parameters))
    replay = Replay(cache_split_dir=cache)
    fps = {r["episode_id"]: float(r.get("fps", 10)) for r in read_index(split_dir=os.path.join(dataset_dir, split))}
    out_root = os.path.join(parameters["storage_dir"], "predictions", "dreamerv3", robot, dataset, tag)
    done = 0
    for episode_id in replay.episodes:
        target = os.path.join(out_root, episode_id)
        if os.path.exists(os.path.join(target, "meta.json")):  # written last: marks a finished episode
            continue
        start = time.time()
        ep = replay.episode(episode_id)
        rate = fps.get(episode_id, 10.0)
        horizon = window_frames(fps=rate, horizon_s=horizon_s, n_frames=len(ep["image"]), context=context)
        if horizon == 0:
            continue
        frames, actions = ep["image"][:context + horizon], ep["action"][:context + horizon]
        os.makedirs(target, exist_ok=True)
        for k in range(samples):
            p = imagine(wm=wm, context=frames[:context], actions=actions, stats=stats, seed=seed + k)
            suffix = "" if k == 0 else f"_s{k}"
            np.save(os.path.join(target, f"pred_features{suffix}.tmp.npy"), p["feat"])
            os.replace(os.path.join(target, f"pred_features{suffix}.tmp.npy"),
                       os.path.join(target, f"pred_features{suffix}.npy"))
            iio.mimwrite(os.path.join(target, f"pred{suffix}.tmp.mp4"), list(p["frames"]), fps=rate,
                         macro_block_size=1, codec="libx264", ffmpeg_params=["-crf", "12"])
            os.replace(os.path.join(target, f"pred{suffix}.tmp.mp4"), os.path.join(target, f"pred{suffix}.mp4"))
        np.save(os.path.join(target, "frame_index.npy"), np.arange(context, context + horizon))
        meta = {"model": "dreamerv3", "context_frames": context, "horizon_frames": horizon, "horizon_s": horizon_s,
                "fps": rate, "seed": seed, "samples": samples, "checkpoint": os.path.join(model_dir, "best.pt"),
                "image_size": s["image_size"], "wall_s": round(time.time() - start, 2)}
        with open(os.path.join(target, "meta.json.tmp"), "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(os.path.join(target, "meta.json.tmp"), os.path.join(target, "meta.json"))
        done += 1
    log_info(f"dreamerv3 {robot}/{dataset}/{split}: predicted {done} episodes into {out_root}", parameters=parameters)


if __name__ == "__main__":
    main()
