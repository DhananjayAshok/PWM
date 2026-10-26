"""Rollout evaluation of one DreamDojo run, reported next to the validation loss in the sweep table:
(a) free-running error of the first 4.8 s (from recorded frame 0, 1 context frame, seed 0) on fixed val episodes:
    latent MSE (generated latents vs VAE latents of the real frames) and LPIPS (decoded frames vs real frames);
(b) action sensitivity: the same error when each episode is driven by another episode's actions (episode k gets the
    actions of episode k+1 of the fixed list). A world model that follows its actions gets clearly worse with the wrong
    ones: sensitivity = shuffled error / true error (latent MSE).

    python -m pwm.wm.dreamdojo.rollout_eval --robot <r> --run_dir <dir> --dataset <data>/<r>/nominal --out <dir> \
        [--shard i/n]
writes <out>/plain/<episode_id>.json (DreamDojo's guidance-0 sampler) and <out>/plain/eval_info.json; `summarise` (used by
the sweep report) averages the episode files.
"""
import glob
import json
import os
import time

import click
import numpy as np

HORIZON_S = 4.8
N_EPISODES = 32
VARIANT = "plain"           # result directory name (kept from when sampler variants existed)
INFO_FILE = "eval_info.json"


def episode_files(*, variant_dir: str) -> list[str]:
    return sorted(f for f in glob.glob(os.path.join(variant_dir, "*.json"))
                  if os.path.basename(f) not in (INFO_FILE, "variant.json"))


def fixed_episodes(*, episodes: list[dict], n: int, commands_per_frame: int) -> list[dict]:
    """The first n val episodes by id that cover the whole window, topped up with the rest; deterministic."""
    eps = sorted(episodes, key=lambda e: e["episode_id"])
    covers = lambda e: e["steps"] // commands_per_frame >= round(HORIZON_S * e["fps"])
    return ([e for e in eps if covers(e)] + [e for e in eps if not covers(e)])[:n]


def summarise(*, variant_dir: str) -> dict | None:
    rows = [json.load(open(f)) for f in episode_files(variant_dir=variant_dir)]
    if not rows:
        return None
    m = lambda k: float(np.mean([r[k] for r in rows]))
    out = {"episodes": len(rows), "latent_mse": m("latent_mse"), "lpips": m("lpips"),
           "latent_mse_shuffled": m("latent_mse_shuffled"), "lpips_shuffled": m("lpips_shuffled")}
    out["sensitivity"] = out["latent_mse_shuffled"] / max(out["latent_mse"], 1e-12)
    out["sensitivity_lpips"] = out["lpips_shuffled"] / max(out["lpips"], 1e-12)
    return out


@click.command()
@click.option("--robot", required=True)
@click.option("--run_dir", required=True)
@click.option("--weights", default="best", help="best | latest | path")
@click.option("--dataset", required=True)
@click.option("--out", required=True, help="Directory for plain/<episode_id>.json")
@click.option("--n_episodes", default=N_EPISODES, type=int)
@click.option("--shard", default="0/1")
def main(robot, run_dir, weights, dataset, out, n_episodes, shard):
    import torch
    from pwm.utils import load_parameters, log_info
    parameters = load_parameters()
    os.environ.setdefault("TORCH_HOME", os.path.join(parameters["storage_dir"], "cache", "torch"))  # LPIPS weights
    import lpips
    from pwm.wm.dreamdojo import data as D
    from pwm.wm.dreamdojo import model as M
    from pwm.wm.dreamdojo.predict import chunk_frame_index, encode_real, load_for_sampling, pad_features, rollout

    cfg = parameters["dreamdojo"]
    rcfg, size = cfg["robots"][robot], tuple(cfg["resolution"])
    weights = {"best": os.path.join(run_dir, "best_ema_bf16.pt"),
               "latest": os.path.join(run_dir, "ema_bf16_latest.pt")}.get(weights, weights)
    norm = D.ActionNormaliser.load(path=os.path.join(run_dir, "action_stats.json"), robot_cfg=rcfg)
    eps = fixed_episodes(episodes=D.read_episodes(split_dirs=[os.path.join(dataset, "val")]), n=n_episodes,
                         commands_per_frame=rcfg["commands_per_frame"])
    vdir = os.path.join(out, VARIANT)
    os.makedirs(vdir, exist_ok=True)
    info = os.path.join(vdir, INFO_FILE)
    with open(info + ".tmp", "w") as f:
        json.dump({"weights": weights, "episodes": [e["episode_id"] for e in eps]}, f)
    os.replace(info + ".tmp", info)
    model = load_for_sampling(parameters=parameters, weights=weights)
    lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    i, n = (int(x) for x in shard.split("/"))

    def actions_of(e, n_tr):
        npz = np.load(e["path"] + ".npz", allow_pickle=True)
        k = D.n_recorded_transitions(n_frames=n_tr + 1, steps=int(npz["action_cmd"].shape[0]), robot_cfg=rcfg)
        return pad_features(features=D.packed_actions(npz=npz, robot_cfg=rcfg, n_transitions=k), rcfg=rcfg, n=n_tr)

    @torch.no_grad()
    def errors(pred, pred_lat, real, real_lat, real_idx, horizon):
        idx = chunk_frame_index(chunks=len(pred_lat) // 3)
        keep = idx <= horizon
        pos = {int(f): j for j, f in enumerate(real_idx)}
        pairs = [(j, pos[int(f)]) for j, f in enumerate(idx) if keep[j] and int(f) in pos]
        mse = float(np.mean([np.mean((pred_lat[a].astype(np.float32) - real_lat[b].astype(np.float32)) ** 2)
                             for a, b in pairs]))
        x = torch.from_numpy(pred[1:horizon + 1]).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
        y = torch.from_numpy(real[1:horizon + 1]).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
        d = torch.cat([lp(x[s:s + 8], y[s:s + 8]).flatten() for s in range(0, len(x), 8)])
        return mse, float(d.mean())

    for k, e in enumerate(eps):
        if k % n != i or os.path.exists(os.path.join(vdir, e["episode_id"] + ".json")):
            continue
        t0 = time.time()
        real = D.decode_frames(path=e["path"], size=size)
        horizon = min(int(round(HORIZON_S * e["fps"])), len(real) - 1)
        n_tr = int(np.ceil(horizon / M.N_CHUNK)) * M.N_CHUNK
        real_lat, real_idx = encode_real(model=model, frames=real[: n_tr + 1] if len(real) > n_tr else real)
        res = {"episode_id": e["episode_id"], "horizon_frames": horizon}
        for tag, src in (("", e), ("_shuffled", eps[(k + 1) % len(eps)])):
            acts = norm.vectors(actions_of(src, n_tr))
            pred, pred_lat = rollout(model=model, frame0=real[0], actions=acts, seed=0,
                                     num_steps=cfg["predict"]["num_steps"])
            res["latent_mse" + tag], res["lpips" + tag] = errors(pred, pred_lat, real, real_lat, real_idx, horizon)
        res["wall_s"] = round(time.time() - t0, 1)
        path = os.path.join(vdir, e["episode_id"] + ".json")
        with open(path + ".tmp", "w") as f:
            json.dump(res, f)
        os.replace(path + ".tmp", path)
        log_info(f"{e['episode_id']}: {res}", parameters=parameters)


if __name__ == "__main__":
    main()
