"""Data for V-JEPA 2-AC: episodes of a PWM dataset (plans/data_format.md) as windows of `frames_per_clip` model steps.

A model step is `frame_stride` recorded frames (configs/world_models.yaml, vjepa2.robots), i.e.
`c = control_steps_per_frame * frame_stride` control steps; the action of a step is the c commands (`action_cmd`) of
that step, stacked. The state input is the measured state at the window start, repeated for every step: world models
see only the start state and the commands, never measured future state.

Frames are decoded once into a cache of model-resolution frames per episode,
`<dataset>/derived/vjepa2-<hash>/frames/<split>/<episode_id>.npy` (uint8, N x crop x crop x 3), built by any number of
workers (claim files; restart-safe), and memory-mapped by the training data loader.
"""
import hashlib
import json
import os
import random
import time
from typing import Any

import click
import numpy as np
import torch

from pwm.utils import load_parameters, log_info, log_warn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def robot_cfg(*, parameters: dict[str, Any], robot: str) -> dict[str, Any]:
    """Merged vjepa2 settings for one robot: the robot entry overrides the shared ones."""
    cfg = dict(parameters["vjepa2"])
    cfg.update(cfg.pop("robots")[robot])
    cfg["robot"] = robot
    cfg["control_steps_per_step"] = cfg["control_steps_per_frame"] * cfg["frame_stride"]
    return cfg


def dataset_root(*, parameters: dict[str, Any], robot: str, dataset: str) -> str:
    return os.path.join(parameters["data_dir"], robot, dataset)


def cache_dir(*, parameters: dict[str, Any], robot: str, dataset: str) -> str:
    """Frame cache of one dataset; the name hashes everything that determines its content."""
    cfg = robot_cfg(parameters=parameters, robot=robot)
    key = json.dumps({"crop": cfg["crop_size"], "stride": cfg["frame_stride"], "v": 1}, sort_keys=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:10]
    return os.path.join(dataset_root(parameters=parameters, robot=robot, dataset=dataset), "derived", f"vjepa2-{h}")


def read_episodes(*, root: str, split: str) -> list[dict[str, Any]]:
    path = os.path.join(root, split, "episodes.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return list({r["episode_id"]: r for r in rows}.values())  # last line of a re-written episode wins


def decode_frames(*, mp4: str, frame_stride: int, crop: int) -> np.ndarray:
    """Every frame_stride-th recorded frame, resized (aspect not kept: the same squash at train and eval time)."""
    import av
    import cv2
    out = []
    with av.open(mp4) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i % frame_stride == 0:
                out.append(cv2.resize(frame.to_ndarray(format="rgb24"), (crop, crop), interpolation=cv2.INTER_AREA))
    return np.stack(out)


def _claim(path: str, stale_s: float = 1800) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(path) > stale_s:  # a dead worker's claim
                os.utime(path)
                return True
        except OSError:
            pass
        return False


def build_frame_cache(*, parameters: dict[str, Any], robot: str, dataset: str, splits: list[str]) -> int:
    """Decode every episode of the splits into the frame cache; skips cached ones. Returns the number written."""
    cfg = robot_cfg(parameters=parameters, robot=robot)
    root = dataset_root(parameters=parameters, robot=robot, dataset=dataset)
    cdir = cache_dir(parameters=parameters, robot=robot, dataset=dataset)
    written = 0
    for split in splits:
        out_dir = os.path.join(cdir, "frames", split)
        claims = os.path.join(cdir, "claims", split)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(claims, exist_ok=True)
        episodes = read_episodes(root=root, split=split)
        random.Random(os.getpid() ^ int(time.time())).shuffle(episodes)  # workers start on different episodes
        for ep in episodes:
            eid = ep["episode_id"]
            final = os.path.join(out_dir, eid + ".npy")
            if os.path.exists(final) or not _claim(os.path.join(claims, eid)):
                continue
            frames = decode_frames(mp4=os.path.join(root, split, eid + ".mp4"), frame_stride=cfg["frame_stride"],
                                   crop=cfg["crop_size"])
            tmp = final + f".tmp{os.getpid()}.npy"
            np.save(tmp, frames)
            os.replace(tmp, final)
            written += 1
    log_info(f"[vjepa2] frame cache {cdir}: {written} episodes written", parameters=parameters)
    return written


class Episode:
    """One cached episode: frames (memory-mapped), stacked commands per model step, states per control step."""

    def __init__(self, *, root: str, cdir: str, split: str, row: dict[str, Any], cfg: dict[str, Any]):
        self.row, self.c = row, cfg["control_steps_per_step"]
        self.frames_path = os.path.join(cdir, "frames", split, row["episode_id"] + ".npy")
        with np.load(os.path.join(root, split, row["episode_id"] + ".npz"), allow_pickle=False) as z:
            cmd = np.asarray(z["action_cmd"], np.float32)
            self.state = np.asarray(z["state"], np.float32)
        self.frames = None
        n_cached = np.load(self.frames_path, mmap_mode="r").shape[0]  # header only
        # frame k is the observation before model step k; usable frames: those cached that have the commands after them
        n_steps = min(len(cmd) // self.c, n_cached - 1)
        self.actions = cmd[: n_steps * self.c].reshape(n_steps, -1)  # (n_steps, c * A)
        self.n_frames = n_steps + 1

    def window(self, *, start: int, length: int):
        if self.frames is None:
            self.frames = np.load(self.frames_path, mmap_mode="r")
        assert start + length <= self.n_frames, (self.row["episode_id"], start, length, self.n_frames)
        frames = np.asarray(self.frames[start : start + length])
        actions = self.actions[start : start + length - 1]
        state0 = self.state[start * self.c]
        return frames, actions, np.repeat(state0[None], length, axis=0)


def load_episodes(*, parameters: dict[str, Any], robot: str, dataset: str, split: str, min_frames: int) -> list[Episode]:
    cfg = robot_cfg(parameters=parameters, robot=robot)
    root = dataset_root(parameters=parameters, robot=robot, dataset=dataset)
    cdir = cache_dir(parameters=parameters, robot=robot, dataset=dataset)
    eps, skipped = [], 0
    for row in read_episodes(root=root, split=split):
        if not os.path.exists(os.path.join(cdir, "frames", split, row["episode_id"] + ".npy")):
            skipped += 1
            continue
        ep = Episode(root=root, cdir=cdir, split=split, row=row, cfg=cfg)
        if ep.n_frames >= min_frames:
            eps.append(ep)
    if skipped:
        log_warn(f"[vjepa2] {split}: {skipped} episodes not in the frame cache yet (skipped)", parameters=parameters)
    return eps


def to_model_input(frames: np.ndarray) -> torch.Tensor:
    """uint8 (..., H, W, 3) -> float, ImageNet-normalised, (..., 3, H, W)."""
    x = (np.asarray(frames, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.moveaxis(x, -1, -3).copy())


class Windows(torch.utils.data.IterableDataset):
    """Infinite random windows (episodes weighted by length). The stream is a deterministic function of
    (seed, rank, worker, resume_step), so a resumed run is reproducible from its resume point."""

    def __init__(self, *, episodes: list[Episode], length: int, seed: int, rank: int, resume_step: int):
        self.episodes, self.length = episodes, length
        self.seed, self.rank, self.resume_step = seed, rank, resume_step
        weights = np.array([e.n_frames - length + 1 for e in episodes], np.float64)
        self.p = weights / weights.sum()

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng([self.seed, self.rank, wid, self.resume_step])
        while True:
            ep = self.episodes[rng.choice(len(self.episodes), p=self.p)]
            start = int(rng.integers(0, ep.n_frames - self.length + 1))
            frames, actions, states = ep.window(start=start, length=self.length)
            yield to_model_input(frames), torch.from_numpy(actions), torch.from_numpy(states)


def val_windows(*, episodes: list[Episode], length: int, max_windows: int) -> list[tuple[int, int]]:
    """Fixed validation windows: evenly spaced starts in every episode, (episode index, start), capped deterministically."""
    out = []
    for i, ep in enumerate(episodes):
        n_start = ep.n_frames - length + 1
        for start in range(0, n_start, max(1, length // 2)):
            out.append((i, start))
    if len(out) > max_windows:
        idx = np.random.default_rng(0).choice(len(out), max_windows, replace=False)
        out = [out[i] for i in sorted(idx)]
    return out


@click.group()
def main():
    pass


@main.command("build-cache")
@click.option("--robot", required=True)
@click.option("--dataset", required=True)
@click.option("--splits", default="train,val,test", help="Comma-separated")
def build_cache_cmd(robot, dataset, splits):
    """Decode a dataset's episodes into the V-JEPA frame cache (run several in parallel to go faster)."""
    parameters = load_parameters()
    build_frame_cache(parameters=parameters, robot=robot, dataset=dataset, splits=splits.split(","))


if __name__ == "__main__":
    main()
