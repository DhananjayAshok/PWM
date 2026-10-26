"""Training / validation clips for DreamDojo from PWM episodes (plans/data_format.md).

A clip is `clip_frames` consecutive recorded frames (1 conditioning + 12) and one 384-d action vector per transition.
Transition k (recorded frame k -> k+1) covers control steps [k*s, (k+1)*s), s = commands_per_frame. Its native action
features are written into the robot's slot (configs/world_models.yaml):
  stack      concat(action_cmd[k*s : (k+1)*s])                       (delta-command robots: EE deltas + gripper)
  net_delta  action_cmd[(k+1)*s - 1] - action_cmd[k*s - 1]            (absolute joint targets; action_cmd[-1] is the
                                                                       start state, whose layout equals the command's)
then normalised per dimension with mean / std over the train split. Everything else in the vector is zero.
"""
import json
import os
import random
import time

import av
import cv2
import numpy as np
import torch

from pwm.wm.dreamdojo.runtime import ACTION_DIM


def read_episodes(*, split_dirs: list[str]) -> list[dict]:
    """episodes.jsonl rows of every split directory, with `path` = <split_dir>/<episode_id>."""
    rows = []
    for d in split_dirs:
        index = os.path.join(d, "episodes.jsonl")
        if not os.path.exists(index):
            continue
        with open(index) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["path"] = os.path.join(d, r["episode_id"])
                    rows.append(r)
    return rows


def decode_frames(*, path: str, size: tuple[int, int]) -> np.ndarray:
    """mp4 -> (n, H, W, 3) uint8 at `size` (H, W)."""
    with av.open(path + ".mp4") as c:
        c.streams.video[0].thread_type = "AUTO"
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    out = np.stack(frames)
    if out.shape[1:3] != tuple(size):
        out = np.stack([cv2.resize(f, (size[1], size[0]), interpolation=cv2.INTER_AREA) for f in out])
    return out


def packed_actions(*, npz: dict, robot_cfg: dict, n_transitions: int) -> np.ndarray:
    """(n_transitions, width) native action features, unnormalised."""
    cmd = np.asarray(npz["action_cmd"], np.float32)
    s = robot_cfg["commands_per_frame"]
    if robot_cfg["packing"] == "stack":
        return cmd[: n_transitions * s].reshape(n_transitions, -1)
    if robot_cfg["packing"] == "net_delta":
        prev = np.concatenate([np.asarray(npz["state"], np.float32)[:1, : cmd.shape[1]], cmd], 0)  # prev[t+1] = cmd[t]
        ends = np.arange(1, n_transitions + 1) * s  # index into prev of cmd[(k+1)s - 1]
        return prev[ends] - prev[ends - s]
    raise ValueError(f"unknown packing {robot_cfg['packing']}")


def n_recorded_transitions(*, n_frames: int, steps: int, robot_cfg: dict) -> int:
    return min(n_frames - 1, steps // robot_cfg["commands_per_frame"])


class ActionNormaliser:
    def __init__(self, *, mean: np.ndarray, std: np.ndarray, robot_cfg: dict):
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.start, self.end = robot_cfg["slot"]
        assert self.end - self.start == len(self.mean), f"slot {robot_cfg['slot']} does not fit {len(self.mean)} dims"

    def vectors(self, features: np.ndarray) -> np.ndarray:
        """(n, width) raw features -> (n, 384) action vectors."""
        out = np.zeros((len(features), ACTION_DIM), np.float32)
        out[:, self.start:self.end] = np.clip((features - self.mean) / self.std, -10, 10)
        return out

    def save(self, path: str) -> None:
        with open(path + ".tmp", "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, *, path: str, robot_cfg: dict) -> "ActionNormaliser":
        with open(path) as f:
            d = json.load(f)
        return cls(mean=d["mean"], std=d["std"], robot_cfg=robot_cfg)

    @classmethod
    def fit(cls, *, episodes: list[dict], robot_cfg: dict) -> "ActionNormaliser":
        feats = []
        for e in episodes:
            npz = np.load(e["path"] + ".npz", allow_pickle=True)
            n_frames = e["steps"] // robot_cfg["commands_per_frame"] + 1
            n = n_recorded_transitions(n_frames=n_frames, steps=e["steps"], robot_cfg=robot_cfg)
            if n > 0:
                feats.append(packed_actions(npz=npz, robot_cfg=robot_cfg, n_transitions=n))
        x = np.concatenate(feats, 0)
        std = x.std(0)
        std[std < 1e-6] = 1.0
        return cls(mean=x.mean(0), std=std, robot_cfg=robot_cfg)


def episode_clips(*, episode: dict, normaliser: ActionNormaliser, robot_cfg: dict, n: int, size, starts=None):
    """Decode one episode once; yield (frames (n,H,W,3) uint8, actions (n-1, 384) float32) for the given starts
    (default: none; caller picks)."""
    frames = decode_frames(path=episode["path"], size=size)
    npz = np.load(episode["path"] + ".npz", allow_pickle=True)
    n_tr = n_recorded_transitions(n_frames=len(frames), steps=int(npz["action_cmd"].shape[0]), robot_cfg=robot_cfg)
    acts = normaliser.vectors(packed_actions(npz=npz, robot_cfg=robot_cfg, n_transitions=n_tr))
    for s in starts:
        yield frames[s:s + n], acts[s:s + n - 1]


class TrainClips(torch.utils.data.IterableDataset):
    """Infinite random clips from the train episodes, episodes drawn proportional to their number of clip starts.
    Each worker decodes one episode at a time and yields `per_decode` windows through a shuffle buffer."""

    def __init__(self, *, episodes, normaliser, robot_cfg, n, size, per_decode=4, buffer=64, seed=0):
        self.episodes = [e for e in episodes if e["steps"] // robot_cfg["commands_per_frame"] + 1 >= n]
        self.normaliser, self.robot_cfg, self.n, self.size = normaliser, robot_cfg, n, size
        self.per_decode, self.buffer, self.seed = per_decode, buffer, seed
        self.weights = np.array([e["steps"] // robot_cfg["commands_per_frame"] + 1 - n + 1 for e in self.episodes], float)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rank = int(os.environ.get("RANK", 0))
        rng = random.Random(hash((self.seed, rank, info.id if info else 0, time.time_ns())))
        buf = []
        while True:
            e = rng.choices(self.episodes, weights=self.weights)[0]
            try:
                n_frames = e["steps"] // self.robot_cfg["commands_per_frame"] + 1
                starts = [rng.randrange(0, max(1, n_frames - self.n + 1)) for _ in range(self.per_decode)]
                for frames, acts in episode_clips(episode=e, normaliser=self.normaliser, robot_cfg=self.robot_cfg,
                                                  n=self.n, size=self.size, starts=starts):
                    if len(frames) == self.n and len(acts) == self.n - 1:
                        buf.append((frames.copy(), acts.copy()))
            except Exception as ex:  # a corrupt episode must not kill training
                print(f"[dreamdojo.data] skip {e['episode_id']}: {ex!r}", flush=True)
                continue
            while len(buf) >= self.buffer:
                yield buf.pop(rng.randrange(len(buf)))


def fixed_val_clips(*, episodes, normaliser, robot_cfg, n, size, count: int):
    """Deterministic validation clips: episodes sorted by id, starts spread evenly; at most `count` clips."""
    eps = sorted((e for e in episodes if e["steps"] // robot_cfg["commands_per_frame"] + 1 >= n),
                 key=lambda e: e["episode_id"])
    if not eps:
        return []
    per = max(1, int(np.ceil(count / len(eps))))
    clips = []
    for e in eps:
        n_frames = e["steps"] // robot_cfg["commands_per_frame"] + 1
        starts = np.linspace(0, n_frames - n, per).round().astype(int).tolist()
        for frames, acts in episode_clips(episode=e, normaliser=normaliser, robot_cfg=robot_cfg, n=n, size=size,
                                          starts=sorted(set(starts))):
            if len(frames) == n:
                clips.append((frames, acts))
        if len(clips) >= count:
            break
    return clips[:count]


def worker_init(_):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
