"""DreamDojo predictions and real-frame features in the layout of plans/predictions_format.md.

`predict`: free-running prediction of the first `horizon_s` of each episode from its recorded frame 0 (1 context
frame) and the commanded actions, in chunks of 12 frames; each chunk is conditioned on the last predicted frame, seed 0.
Writes per episode <out>/<episode_id>/{pred.mp4, pred_features.npy, frame_index.npy, meta.json}: pred.mp4 holds the
context frame then the predicted frames (horizon + 1 frames, recording fps); pred_features are the generated video-VAE
latents before decoding, one per latent frame after the conditioning one (latent j of a chunk covers chunk frames
4j-3 .. 4j; frame_index = the last recorded frame it covers).
`encode`: VAE latents of the REAL frames in the same space and layout (13-frame windows every 12 frames), for training
the state probes: <out>/<episode_id>.npz with real_features, frame_index.

    python -m pwm.wm.dreamdojo.predict predict --robot <r> --dataset <data>/<r>/<dataset> --split test \
        --weights <run_dir>/best_ema_bf16.pt --stats <run_dir>/action_stats.json --checkpoint_tag <tag>
    python -m pwm.wm.dreamdojo.predict encode --robot <r> --dataset <data>/<r>/<dataset> --split train --max_episodes 2000
"""
import json
import os
import time

import click
import imageio.v2 as iio
import numpy as np
import torch

from pwm.utils import load_parameters, log_info
from pwm.wm.dreamdojo import data as D
from pwm.wm.dreamdojo import model as M
from pwm.wm.dreamdojo.assets import init_weights_path

HORIZON_S = 4.8


def _to_uint8(video: torch.Tensor) -> np.ndarray:
    """(1, C, T, H, W) in [-1, 1] -> (T, H, W, 3) uint8."""
    v = torch.clamp((video[0].float() + 1) / 2, 0, 1)
    return (v * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


def load_for_sampling(*, parameters, weights: str):
    """The model with `weights`, in eval mode."""
    model = M.load(parameters=parameters, weights=weights)
    model.net.eval()
    return model


@torch.no_grad()
def rollout(*, model, frame0: np.ndarray, actions: np.ndarray, seed: int, num_steps: int):
    """frame0 (H, W, 3) uint8, actions (12 K, 384) -> frames (12 K + 1, H, W, 3) uint8, latents (3 K, C, h, w) fp16."""
    assert len(actions) % M.N_CHUNK == 0
    frames, latents, cond = [frame0], [], frame0
    for c in range(len(actions) // M.N_CHUNK):
        vid = np.zeros((M.N_CHUNK + 1,) + frame0.shape, np.uint8)
        vid[0] = cond
        vid_t = torch.from_numpy(vid)[None].cuda()
        act = torch.from_numpy(actions[c * M.N_CHUNK:(c + 1) * M.N_CHUNK])[None].cuda()
        lat = M.sample(model=model, batch=M.inference_batch(model=model, frames=vid_t, actions=act),
                       seed=seed + c * M.N_CHUNK, num_steps=num_steps)
        out = _to_uint8(model.decode(lat))
        frames.extend(out[1:])
        latents.append(lat[0, :, 1:].permute(1, 0, 2, 3).to(torch.float16).cpu().numpy())
        cond = out[-1]
    return np.stack(frames), np.concatenate(latents, 0)


def chunk_frame_index(*, chunks: int) -> np.ndarray:
    """Last recorded frame covered by each non-conditioning latent of `chunks` consecutive 12-frame chunks from frame 0."""
    return np.array([c * M.N_CHUNK + 4 * j for c in range(chunks) for j in (1, 2, 3)])


@torch.no_grad()
def encode_real(*, model, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All recorded frames -> (K, C, h, w) fp16 latents and their frame_index: latent 0 of the first window (frame 0
    alone) and latents 1..3 of each 13-frame window starting at 0, 12, 24, ... (the windows the predictions use)."""
    feats, index = [], []
    for w in range(0, max(1, (len(frames) - 1) // M.N_CHUNK)):
        s = w * M.N_CHUNK
        win = frames[s:s + M.N_CHUNK + 1]
        if len(win) < M.N_CHUNK + 1:
            break
        x = (torch.from_numpy(win).cuda().permute(3, 0, 1, 2)[None].float() / 127.5 - 1).to(torch.bfloat16)
        lat = model.encode(x)[0].permute(1, 0, 2, 3).to(torch.float16).cpu().numpy()  # (4, C, h, w)
        keep = range(4) if w == 0 else range(1, 4)
        feats.extend(lat[j] for j in keep)
        index.extend(s + 4 * j for j in keep)
    return np.stack(feats), np.array(index)


def pad_features(*, features: np.ndarray, rcfg: dict, n: int) -> np.ndarray:
    """Extend raw action features to n transitions with 'hold still' commands (zero motion; for stacked 7-D commands the
    gripper keeps its last value), so a window longer than the episode can still be predicted in whole chunks."""
    if len(features) >= n:
        return features[:n]
    pad = np.zeros((n - len(features), features.shape[1]), np.float32)
    if rcfg["packing"] == "stack" and len(features):
        width = features.shape[1] // rcfg["commands_per_frame"]
        pad[:, width - 1::width] = features[-1, width - 1::width]
    return np.concatenate([features, pad], 0)


def _episodes(*, dataset: str, split: str, episode_list: str) -> list[dict]:
    eps = D.read_episodes(split_dirs=[os.path.join(dataset, split)])
    if episode_list != "none":
        keep = {os.path.basename(p.strip()) for p in open(episode_list) if p.strip()}
        eps = [e for e in eps if e["episode_id"] in keep]
    return sorted(eps, key=lambda e: e["episode_id"])


@click.group()
def cli():
    pass


@cli.command()
@click.option("--robot", required=True)
@click.option("--dataset", required=True, help="Dataset root <data_dir>/<robot>/<dataset>")
@click.option("--split", default="test")
@click.option("--weights", required=True, help="Inference-format weights (a run's best_ema_bf16.pt)")
@click.option("--stats", required=True, help="The run's action_stats.json")
@click.option("--checkpoint_tag", required=True)
@click.option("--episodes", "episode_list", default="none", help="Optional file of episode ids/paths to restrict to")
@click.option("--out", default="none", help="none = <storage>/predictions/dreamdojo/<robot>/<dataset>/<checkpoint_tag>")
@click.option("--seed", default=0, type=int)
@click.option("--shard", default="0/1", help="i/n: this process handles every n-th episode, offset i")
def predict(robot, dataset, split, weights, stats, checkpoint_tag, episode_list, out, seed, shard):
    parameters = load_parameters()
    cfg = parameters["dreamdojo"]
    rcfg, size = cfg["robots"][robot], tuple(cfg["resolution"])
    if out == "none":
        out = os.path.join(parameters["storage_dir"], "predictions", "dreamdojo", robot, os.path.basename(dataset.rstrip("/")),
                           checkpoint_tag)
    norm = D.ActionNormaliser.load(path=stats, robot_cfg=rcfg)
    i, n = (int(x) for x in shard.split("/"))
    eps = _episodes(dataset=dataset, split=split, episode_list=episode_list)[i::n]
    model = load_for_sampling(parameters=parameters, weights=weights)
    for e in eps:
        edir = os.path.join(out, e["episode_id"])
        if os.path.exists(os.path.join(edir, "meta.json")):
            continue
        t0 = time.time()
        frames = D.decode_frames(path=e["path"], size=size)
        npz = np.load(e["path"] + ".npz", allow_pickle=True)
        horizon = min(int(round(HORIZON_S * e["fps"])), len(frames) - 1)
        chunks = int(np.ceil(horizon / M.N_CHUNK))
        n_tr = D.n_recorded_transitions(n_frames=len(frames), steps=int(npz["action_cmd"].shape[0]), robot_cfg=rcfg)
        feats = pad_features(features=D.packed_actions(npz=npz, robot_cfg=rcfg, n_transitions=n_tr), rcfg=rcfg,
                             n=chunks * M.N_CHUNK)
        pred, lat = rollout(model=model, frame0=frames[0], actions=norm.vectors(feats), seed=seed,
                            num_steps=cfg["predict"]["num_steps"])
        index = chunk_frame_index(chunks=chunks)
        keep = index <= horizon
        os.makedirs(edir + ".tmp", exist_ok=True)
        iio.mimwrite(os.path.join(edir + ".tmp", "pred.mp4"), list(pred[: horizon + 1]), fps=e["fps"], macro_block_size=8)
        np.save(os.path.join(edir + ".tmp", "pred_features.npy"), lat[keep])
        np.save(os.path.join(edir + ".tmp", "frame_index.npy"), index[keep])
        meta = {"context_frames": 1, "horizon_frames": horizon, "seed": seed, "checkpoint": weights,
                "checkpoint_tag": checkpoint_tag, "wall_s": round(time.time() - t0, 1), "chunks": chunks,
                "pred_mp4": "frame 0 = the real context frame, frames 1..horizon_frames predicted",
                "features": "Wan2.1 VAE latents (16, H/8, W/8) of the generated video, before decoding"}
        with open(os.path.join(edir + ".tmp", "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if os.path.exists(edir):  # a half-written earlier attempt without meta.json
            import shutil
            shutil.rmtree(edir)
        os.replace(edir + ".tmp", edir)
        log_info(f"{e['episode_id']}: {horizon} frames in {meta['wall_s']} s", parameters=parameters)


@cli.command()
@click.option("--robot", required=True)
@click.option("--dataset", required=True, help="Dataset root <data_dir>/<robot>/<dataset>")
@click.option("--split", default="train")
@click.option("--checkpoint_tag", default="vae", help="The VAE is frozen, so real features do not depend on the run")
@click.option("--max_episodes", default=0, type=int, help="Deterministic sample of at most this many (0 = all)")
@click.option("--out", default="none",
              help="none = <storage>/predictions/dreamdojo/<robot>/features/<checkpoint_tag>/<dataset>/<split>")
@click.option("--shard", default="0/1")
def encode(robot, dataset, split, checkpoint_tag, max_episodes, out, shard):
    parameters = load_parameters()
    size = tuple(parameters["dreamdojo"]["resolution"])
    if out == "none":
        out = os.path.join(parameters["storage_dir"], "predictions", "dreamdojo", robot, "features", checkpoint_tag,
                           os.path.basename(dataset.rstrip("/")), split)
    os.makedirs(out, exist_ok=True)
    eps = _episodes(dataset=dataset, split=split, episode_list="none")
    if max_episodes and len(eps) > max_episodes:
        eps = eps[:: int(np.ceil(len(eps) / max_episodes))]
    i, n = (int(x) for x in shard.split("/"))
    model = M.load(parameters=parameters, weights=init_weights_path(parameters))
    for e in eps[i::n]:
        path = os.path.join(out, e["episode_id"] + ".npz")
        if os.path.exists(path):
            continue
        feats, index = encode_real(model=model, frames=D.decode_frames(path=e["path"], size=size))
        np.savez(path[:-4] + ".tmp.npz", real_features=feats, frame_index=index)
        os.replace(path[:-4] + ".tmp.npz", path)
    log_info(f"encoded {len(eps[i::n])} episodes of {dataset}/{split} into {out}", parameters=parameters)


if __name__ == "__main__":
    cli()
