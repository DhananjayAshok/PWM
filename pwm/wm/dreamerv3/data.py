"""Offline replay for DreamerV3 from our recorded episodes (plans/data_format.md).

Every episode of a split is decoded once to low resolution and kept in a cache under the dataset's ``derived/``
directory (shards of whole episodes, written atomically). Training then holds a split in memory as one sequence of
frames with episode boundaries, and samples fixed-length windows as the port's replay buffer does: a window may cross
into the next episode, where ``is_first`` restarts the model's state.

Alignment of frames and commands: the npz holds every control step, the mp4 only recorded frames, so
``stride = steps / (frames - 1)`` control steps fall between two recorded frames. The model input at frame ``i`` is the
commands of the transition into it, ``action_cmd[(i-1)*stride : i*stride]`` stacked (zeros at an episode's first frame,
DreamerV3's convention), and ``success`` at frame ``i`` is whether the task was solved during that transition.
"""
import fcntl
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional

import av
import numpy as np

CACHE_VERSION = 1
SHARD_EPISODES = 256


def read_index(*, split_dir: str) -> list[dict[str, Any]]:
    """Finished episodes of one split directory (episodes.jsonl), in file order."""
    path = os.path.join(split_dir, "episodes.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return list({r["episode_id"]: r for r in rows}.values())  # last line wins if an episode was rewritten


def decode_frames(*, path: str, size: int) -> np.ndarray:
    """All frames of an mp4 as (N, size, size, 3) uint8 (area-averaged downscale of the full frame)."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [f.reformat(width=size, height=size, format="rgb24", interpolation="AREA").to_ndarray()
                  for f in container.decode(stream)]
    return np.stack(frames)


def align_episode(*, frames: np.ndarray, action_cmd: np.ndarray, success: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame model inputs: stacked commands into each frame, success per transition, is_first."""
    n, steps = len(frames), len(action_cmd)
    stride = max(1, int(round(steps / max(n - 1, 1))))
    adim = action_cmd.shape[1]
    actions = np.zeros((n, stride * adim), np.float32)
    succ = np.zeros(n, np.float32)
    for i in range(1, n):
        chunk = action_cmd[(i - 1) * stride:i * stride]
        if len(chunk) == 0:
            chunk = action_cmd[-1:]
        if len(chunk) < stride:  # short final transition: repeat its last command
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], stride - len(chunk), 0)])
        actions[i] = chunk.reshape(-1)
        s = success[(i - 1) * stride:i * stride]
        succ[i] = float(np.any(s)) if len(s) else succ[i - 1]
    is_first = np.zeros(n, bool)
    is_first[0] = True
    return {"image": frames, "action": actions, "success": succ, "is_first": is_first, "stride": stride}


def _load_episode(args) -> Optional[dict[str, np.ndarray]]:
    split_dir, episode_id, size = args
    base = os.path.join(split_dir, episode_id)
    try:
        frames = decode_frames(path=base + ".mp4", size=size)
        npz = np.load(base + ".npz", allow_pickle=True)
        cmd = np.asarray(npz["action_cmd"] if "action_cmd" in npz.files else npz["action_env"], np.float32)
        success = np.asarray(npz["success"], np.float32).reshape(-1)
    except Exception as e:  # unreadable episode: skipped, reported by the caller
        return {"error": f"{episode_id}: {e}"}
    out = align_episode(frames=frames, action_cmd=cmd, success=success)
    out["episode_id"] = episode_id
    return out


def cache_dir(*, dataset_dir: str, size: int) -> str:
    key = hashlib.sha1(json.dumps({"v": CACHE_VERSION, "size": size}).encode()).hexdigest()[:8]
    return os.path.join(dataset_dir, "derived", f"dreamerv3-{size}px-{key}")


def build_cache(*, dataset_dir: str, split: str, size: int, workers: int = 16, log=print) -> str:
    """Decode every episode of ``split`` not yet cached into shards; idempotent, restart-safe, and safe for several
    processes at once (an exclusive lock per cache split). Returns the cache dir."""
    out = os.path.join(cache_dir(dataset_dir=dataset_dir, size=size), split)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_cache_locked(dataset_dir=dataset_dir, split=split, size=size, out=out, workers=workers, log=log)


def _build_cache_locked(*, dataset_dir: str, split: str, size: int, out: str, workers: int, log) -> str:
    split_dir = os.path.join(dataset_dir, split)
    episodes = [r["episode_id"] for r in read_index(split_dir=split_dir)]
    done = set()
    for name in sorted(os.listdir(out)):
        if name.endswith(".json") and name.startswith("shard_"):
            with open(os.path.join(out, name)) as f:
                done.update(json.load(f)["episodes"])
    todo = [e for e in episodes if e not in done]
    shard_id = len([n for n in os.listdir(out) if n.startswith("shard_") and n.endswith(".json")])
    log(f"[cache] {split}: {len(episodes)} episodes, {len(done)} cached, {len(todo)} to decode at {size}px")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(todo), SHARD_EPISODES):
            batch = todo[start:start + SHARD_EPISODES]
            results = list(pool.map(_load_episode, [(split_dir, e, size) for e in batch], chunksize=4))
            good = [r for r in results if r is not None and "error" not in r]
            for r in results:
                if r is not None and "error" in r:
                    log(f"[cache] skipped {r['error']}")
            if not good:
                continue
            name = os.path.join(out, f"shard_{shard_id:05d}")
            lengths = np.array([len(r["image"]) for r in good])
            np.savez(name + ".tmp.npz", image=np.concatenate([r["image"] for r in good]),
                     action=np.concatenate([r["action"] for r in good]),
                     success=np.concatenate([r["success"] for r in good]),
                     is_first=np.concatenate([r["is_first"] for r in good]), lengths=lengths)
            os.replace(name + ".tmp.npz", name + ".npz")
            meta = {"episodes": [r["episode_id"] for r in good], "lengths": lengths.tolist(),
                    "strides": [int(r["stride"]) for r in good]}
            with open(name + ".json.tmp", "w") as f:
                json.dump(meta, f)
            os.replace(name + ".json.tmp", name + ".json")  # the json marks the shard complete
            shard_id += 1
            log(f"[cache] {split}: shard {shard_id} ({start + len(batch)}/{len(todo)})")
    return out


class Replay:
    """One split held in memory as a single frame sequence; samples (B, L) windows."""

    def __init__(self, *, cache_split_dir: str, max_episodes: Optional[int] = None):
        parts = {"image": [], "action": [], "success": [], "is_first": []}
        self.episodes, self.lengths = [], []
        for name in sorted(os.listdir(cache_split_dir)):
            if not (name.startswith("shard_") and name.endswith(".json")):
                continue
            with open(os.path.join(cache_split_dir, name)) as f:
                meta = json.load(f)
            z = np.load(os.path.join(cache_split_dir, name[:-5] + ".npz"))
            for k in parts:
                parts[k].append(z[k])
            self.episodes += meta["episodes"]
            self.lengths += meta["lengths"]
            if max_episodes and len(self.episodes) >= max_episodes:
                break
        if not self.episodes:
            raise ValueError(f"no cached episodes in {cache_split_dir}")
        self.data = {k: np.concatenate(v) for k, v in parts.items()}
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.n = len(self.data["image"])
        self.action_dim = self.data["action"].shape[1]
        self.image_shape = self.data["image"].shape[1:]

    def window(self, start: int, length: int) -> dict[str, np.ndarray]:
        sl = slice(start, start + length)
        w = {k: self.data[k][sl] for k in ("image", "action", "success", "is_first")}
        w["is_first"] = w["is_first"].copy()
        w["is_first"][0] = True  # every window starts a fresh model state
        w["action"] = w["action"].copy()
        w["action"][w["is_first"]] = 0.0
        return w

    def sample(self, *, rng: np.random.Generator, batch: int, length: int) -> dict[str, np.ndarray]:
        starts = rng.integers(0, max(self.n - length, 1), size=batch)
        return stack_windows([self.window(int(s), length) for s in starts])

    def fixed_windows(self, *, count: int, length: int, seed: int = 0) -> list[int]:
        """Deterministic window starts for the validation loss."""
        return sorted(np.random.default_rng(seed).integers(0, max(self.n - length, 1), size=count).tolist())

    def episode(self, episode_id: str) -> dict[str, np.ndarray]:
        i = self.episodes.index(episode_id)
        o, n = int(self.offsets[i]), int(self.lengths[i])
        return {k: self.data[k][o:o + n] for k in ("image", "action", "success", "is_first")}


def stack_windows(windows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    batch = {k: np.stack([w[k] for w in windows]) for k in windows[0]}
    batch["reward"] = batch.pop("success")         # the port's reward head learns per-step success
    batch["is_terminal"] = np.zeros_like(batch["reward"])  # offline windows never end in a terminal state
    return batch


def action_stats(*, replay: Replay) -> dict[str, list[float]]:
    a = replay.data["action"][~replay.data["is_first"]]
    return {"mean": a.mean(0).tolist(), "std": np.maximum(a.std(0), 1e-3).tolist()}
