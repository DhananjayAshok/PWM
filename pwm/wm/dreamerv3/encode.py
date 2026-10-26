"""Encode entry (plans/predictions_format.md): posterior latent states of REAL frames, the features the state probe is
trained on, in the same space as the imagined states written by predict.py: (K, D) vectors, the RSSM's stochastic
state flattened and concatenated with its deterministic state. One npz per episode with all recorded frames:
real_features (K, D), frame_index (K,). Runs on the nominal train split (a deterministic sample of ``--max_episodes``),
on calibration pools and on every test split (all episodes unless ``--max_episodes`` is given).

    python -m pwm.wm.dreamerv3.encode --model_dir <dir with best.pt> --robot franka_libero --dataset nominal \
        --split train [--max_episodes 2000] [--checkpoint_tag <tag>]
"""
import os
import zlib

import click
import numpy as np

from pwm.utils import load_parameters, log_info
from pwm.wm.dreamerv3.data import Replay, build_cache
from pwm.wm.dreamerv3.predict import checkpoint_tag_of, encode_real, load_model


@click.command()
@click.option("--model_dir", required=True)
@click.option("--robot", required=True)
@click.option("--dataset", default="nominal")
@click.option("--split", default="train")
@click.option("--checkpoint_tag", default=None, help="Default: the model directory's name")
@click.option("--max_episodes", default=None, type=int, help="Default: 2000 on train, all elsewhere; 0 = all")
def main(model_dir, robot, dataset, split, checkpoint_tag, max_episodes):
    parameters = load_parameters()
    wm, stats, s = load_model(model_dir=model_dir, parameters=parameters)
    tag = checkpoint_tag or checkpoint_tag_of(model_dir)
    dataset_dir = os.path.join(parameters["data_dir"], robot, dataset)
    cache = build_cache(dataset_dir=dataset_dir, split=split, size=s["image_size"],
                        workers=int(os.environ.get("OMP_NUM_THREADS", "8")),
                        log=lambda m: log_info(m, parameters=parameters))
    replay = Replay(cache_split_dir=cache)
    episodes = sorted(replay.episodes, key=lambda e: zlib.crc32(e.encode()))  # deterministic sample
    if max_episodes is None:
        max_episodes = 2000 if split == "train" else 0
    if max_episodes:
        episodes = episodes[:max_episodes]
    out = os.path.join(parameters["storage_dir"], "predictions", "dreamerv3", robot, "features", tag, dataset, split)
    os.makedirs(out, exist_ok=True)
    done = 0
    for episode_id in episodes:
        path = os.path.join(out, episode_id + ".npz")
        if os.path.exists(path):
            continue
        ep = replay.episode(episode_id)
        feat = encode_real(wm=wm, frames=ep["image"], actions=ep["action"], stats=stats)
        np.savez(path[:-4] + ".tmp.npz", real_features=feat, frame_index=np.arange(len(feat)))
        os.replace(path[:-4] + ".tmp.npz", path)
        done += 1
    log_info(f"dreamerv3 encode {robot}/{dataset}/{split}: {done} new episodes in {out}", parameters=parameters)


if __name__ == "__main__":
    main()
