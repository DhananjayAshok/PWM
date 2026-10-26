"""V-JEPA 2-AC entry points of the eval contract (plans/predictions_format.md).

`predict`: for each episode, from the first recorded frame (1 context frame) and the commands of the first 4.8 s,
predicts free-running in feature space (each step from the model's own previous predictions). Writes
`$storage_dir/predictions/vjepa2/<robot>/<dataset>/<tag>/<episode_id>/{pred_features.npy, frame_index.npy, meta.json}`.
`encode`: frozen-encoder features of REAL frames (same space), e.g. the nominal train split or a calibration pool, to
`$storage_dir/predictions/vjepa2/<robot>/features/<tag>/<dataset>/<split>/<episode_id>.npz` (real_features, frame_index).

Features exist per model step (every `frame_stride` recorded frames), so frame_index lists recorded frames
0, s, 2s, ...; the context frame 0 is not part of pred_features. Tokens are pooled to a 4x4 grid by default
(16 x 1408 per frame; `--pool none` keeps all 256 patch tokens, `mean` keeps 1), float16.

    python -m pwm.wm.vjepa2.predict predict --robot franka_libero --dataset nominal --split test --checkpoint <best.pt>
    python -m pwm.wm.vjepa2.predict encode --robot franka_libero --dataset nominal --split train --max_episodes 2000
"""
import json
import os
import time

import click
import numpy as np
import torch
import torch.nn.functional as F

from pwm.utils import load_parameters, log_info
from pwm.wm.vjepa2 import data, model

HORIZON_S = 4.8
CONTEXT_FRAMES = 1


def pool_tokens(h: torch.Tensor, pool: str) -> torch.Tensor:
    """(..., 256, D) -> (..., n_tok, D)."""
    if pool == "none":
        return h
    if pool == "mean":
        return h.mean(-2, keepdim=True)
    lead, d = h.shape[:-2], h.shape[-1]
    g = h.reshape(-1, 16, 16, d).permute(0, 3, 1, 2)
    return F.avg_pool2d(g.float(), 4).permute(0, 2, 3, 1).reshape(*lead, 16, d)


def load_predictor(*, parameters, checkpoint: str, device: str):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder, predictor = model.build(parameters=parameters, action_dim=ck["action_dim"], state_dim=ck["state_dim"],
                                     device=device)
    predictor.load_state_dict(ck["predictor"])
    return encoder, predictor.eval(), ck


def horizon_steps(*, ep: data.Episode, cfg: dict) -> int:
    """Model steps covering the first 4.8 s (or the whole episode if shorter)."""
    horizon_frames = round(HORIZON_S * ep.row["fps"])
    steps = horizon_frames // cfg["frame_stride"]
    return min(steps, ep.n_frames - CONTEXT_FRAMES, model.MAX_FRAMES - CONTEXT_FRAMES)


@torch.no_grad()
def predict_episode(*, encoder, predictor, ep: data.Episode, horizon: int, device: str) -> torch.Tensor:
    """Predicted tokens (horizon, 256, D) of model steps 1..horizon from frame 0 and the commands."""
    frames, actions, states = ep.window(start=0, length=CONTEXT_FRAMES + horizon)
    x = data.to_model_input(frames[:CONTEXT_FRAMES]).unsqueeze(0).to(device)
    a = torch.from_numpy(actions).unsqueeze(0).to(device)
    s = torch.from_numpy(states).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h_ctx = model.encode(encoder, x)
        pred = model.rollout(predictor, h_ctx, a, s, horizon)
    return pred[0]


def predictions_root(*, parameters, robot: str) -> str:
    return os.path.join(parameters["storage_dir"], "predictions", "vjepa2", robot)


@click.group()
def main():
    pass


@main.command("predict")
@click.option("--robot", required=True)
@click.option("--dataset", required=True)
@click.option("--split", default="test")
@click.option("--checkpoint", required=True, help="best.pt of a training run")
@click.option("--tag", default=None, help="Checkpoint tag (default: the run directory's name)")
@click.option("--pool", default="grid4", type=click.Choice(["none", "grid4", "mean"]))
@click.option("--shard", default="0/1", help="i/n: this worker's share of the episodes")
def predict_cmd(robot, dataset, split, checkpoint, tag, pool, shard):
    parameters = load_parameters()
    cfg = data.robot_cfg(parameters=parameters, robot=robot)
    tag = tag or os.path.basename(os.path.dirname(os.path.abspath(checkpoint)))
    i, n = (int(x) for x in shard.split("/"))
    torch.manual_seed(0)
    encoder, predictor, _ = load_predictor(parameters=parameters, checkpoint=checkpoint, device="cuda")
    eps = data.load_episodes(parameters=parameters, robot=robot, dataset=dataset, split=split,
                             min_frames=CONTEXT_FRAMES + 1)
    out_root = os.path.join(predictions_root(parameters=parameters, robot=robot), dataset, tag)
    done = 0
    for ep in eps[i::n]:
        out = os.path.join(out_root, ep.row["episode_id"])
        if os.path.exists(os.path.join(out, "meta.json")):
            continue
        t0 = time.time()
        hz = horizon_steps(ep=ep, cfg=cfg)
        pred = predict_episode(encoder=encoder, predictor=predictor, ep=ep, horizon=hz, device="cuda")
        tmp = out + f".tmp{os.getpid()}"
        os.makedirs(tmp, exist_ok=True)
        np.save(os.path.join(tmp, "pred_features.npy"), pool_tokens(pred, pool).cpu().numpy().astype(np.float16))
        np.save(os.path.join(tmp, "frame_index.npy"), np.arange(1, hz + 1) * cfg["frame_stride"])
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump({"context_frames": CONTEXT_FRAMES, "horizon_frames": int(hz * cfg["frame_stride"]),
                       "model_steps": int(hz), "frame_stride": cfg["frame_stride"], "seed": 0, "pool": pool,
                       "checkpoint": os.path.abspath(checkpoint), "split": split,
                       "wall_s": round(time.time() - t0, 2)}, f)
        if os.path.exists(out):  # a partial earlier attempt
            import shutil
            shutil.rmtree(out)
        os.replace(tmp, out)
        done += 1
    log_info(f"[vjepa2] predictions {out_root}: {done} written", parameters=parameters)


@main.command("encode")
@click.option("--robot", required=True)
@click.option("--dataset", required=True)
@click.option("--split", default="train")
@click.option("--tag", default="frozen-encoder",
              help="The encoder is frozen, so real features are the same for every fine-tuned predictor")
@click.option("--max_episodes", default=0, type=int, help="0: all; otherwise a fixed random sample (seed 0)")
@click.option("--pool", default="grid4", type=click.Choice(["none", "grid4", "mean"]))
@click.option("--shard", default="0/1")
@click.option("--batch", default=64, type=int)
def encode_cmd(robot, dataset, split, tag, max_episodes, pool, shard, batch):
    parameters = load_parameters()
    cfg = data.robot_cfg(parameters=parameters, robot=robot)
    i, n = (int(x) for x in shard.split("/"))
    encoder, _ = model.build(parameters=parameters, action_dim=7, state_dim=7, device="cuda")
    eps = data.load_episodes(parameters=parameters, robot=robot, dataset=dataset, split=split, min_frames=1)
    if max_episodes and len(eps) > max_episodes:
        keep = np.random.default_rng(0).choice(len(eps), max_episodes, replace=False)
        eps = [eps[k] for k in sorted(keep)]
    out = os.path.join(predictions_root(parameters=parameters, robot=robot), "features", tag, dataset, split)
    os.makedirs(out, exist_ok=True)
    done = 0
    for ep in eps[i::n]:
        path = os.path.join(out, ep.row["episode_id"] + ".npz")
        if os.path.exists(path):
            continue
        frames = np.asarray(np.load(ep.frames_path, mmap_mode="r"))[: ep.n_frames]  # every model step
        feats = []
        for b in range(0, len(frames), batch):
            x = data.to_model_input(frames[b : b + batch]).unsqueeze(1).cuda()  # (b, 1, 3, H, W)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encode(encoder, x).view(len(x), model.TOKENS_PER_FRAME, -1)
            feats.append(pool_tokens(h, pool).cpu().numpy().astype(np.float16))
        np.savez(path + f".tmp{os.getpid()}.npz", real_features=np.concatenate(feats),
                 frame_index=np.arange(len(frames)) * cfg["frame_stride"], pool=pool)
        os.replace(path + f".tmp{os.getpid()}.npz", path)
        done += 1
    log_info(f"[vjepa2] real features {out}: {done} written", parameters=parameters)


if __name__ == "__main__":
    main()
