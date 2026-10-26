"""Evaluate one world model on one robot's test sets and write one result file per dataset.

    python -m pwm.eval.panel --model dreamdojo --robot widowx_bridge --tag best --datasets nominal-test,shift-latency-L3

For each dataset (test split, counterfactual groups):
  pixel metrics      pixel models only; DreamDojo at its native resolution, DreamerV3 at both ("pred" and "real")
  state probe error  probe trained on the model's REAL features of the nominal train split (nominal-test) or of the
                     dataset's calibration pool (shift-<s>-L<l> -> calib-<s>-L<l>); floor = probe on real test features
  counterfactual     accuracy over all ordered pairs of every group (model's own feature space)
Results: <results_dir>/panel/<model>/<robot>/<tag>/<dataset>.json
"""
import json
import os

import click
import numpy as np
import torch

from pwm.eval import counterfactual, io, pixel, probe
from pwm.robots import get_robot
from pwm.utils import load_parameters, log_info, log_warn

PIXEL_MODES = {"dreamdojo": ["pred"], "dreamerv3": ["pred", "real"], "vjepa2": []}
STRIDE = {"franka_libero": 2, "gr1_robocasa": 2, "widowx_bridge": 1, "google_fractal": 1}
MAX_PROBE_EPISODES = 2000


def calib_for(dataset):
    return None if dataset == "nominal-test" else dataset.replace("shift-", "calib-", 1)


def load_real(*, parameters, model, robot, tag, dataset, split, rows):
    out = {}
    for r in rows:
        path = io.real_features_path(storage_dir=parameters["storage_dir"], model=model, robot=robot, tag=tag,
                                     dataset=dataset, split=split, episode_id=r["episode_id"])
        if os.path.exists(path):
            d = np.load(path)
            out[r["episode_id"]] = (d["real_features"], d["frame_index"])
    return out


def probe_data(*, parameters, robot, dataset, split, rows, feats, stride):
    X, Y, G = [], [], []
    for r in rows:
        if r["episode_id"] not in feats:
            continue
        f, idx = feats[r["episode_id"]]
        _, npz = io.episode_paths(data_dir=parameters["data_dir"], robot=robot, dataset=dataset, split=split,
                                  episode_id=r["episode_id"])
        targets = probe.targets_at(probe_state=np.load(npz)["probe_state"], frame_index=idx, stride=stride)
        X.append(probe.flatten_features(f).numpy())
        Y.append(targets)
        G += [r["episode_id"]] * len(targets)
    return np.concatenate(X), np.concatenate(Y), np.array(G)


def get_probe(*, parameters, model, robot, tag, train_dataset, train_split, stride):
    path = os.path.join(parameters["storage_dir"], "eval", "probes", model, robot, tag, f"{train_dataset}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    rows = io.episode_rows(data_dir=parameters["data_dir"], robot=robot, dataset=train_dataset, split=train_split)
    rng = np.random.default_rng(0)
    if len(rows) > MAX_PROBE_EPISODES:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), MAX_PROBE_EPISODES, replace=False))]
    feats = load_real(parameters=parameters, model=model, robot=robot, tag=tag, dataset=train_dataset,
                      split=train_split, rows=rows)
    X, Y, G = probe_data(parameters=parameters, robot=robot, dataset=train_dataset, split=train_split, rows=rows,
                         feats=feats, stride=stride)
    log_info(f"probe {model}/{robot}/{train_dataset}: {len(X)} samples from {len(feats)} episodes, dim {X.shape[1]}",
             parameters=parameters)
    fitted = probe.fit_probe(X=X, Y=Y, groups=G, robot=robot)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(fitted, path + ".tmp")
    os.replace(path + ".tmp", path)
    return fitted


def evaluate_dataset(*, parameters, model, robot, tag, dataset, fps):
    stride = STRIDE[robot]
    rows = io.episode_rows(data_dir=parameters["data_dir"], robot=robot, dataset=dataset, split="test")
    calib = calib_for(dataset)
    fitted = get_probe(parameters=parameters, model=model, robot=robot, tag=tag,
                       train_dataset=calib or "nominal", train_split="calib" if calib else "train", stride=stride)
    reals = load_real(parameters=parameters, model=model, robot=robot, tag=tag, dataset=dataset, split="test", rows=rows)
    per_episode, groups = {}, {}
    for r in rows:
        pdir = io.prediction_dir(storage_dir=parameters["storage_dir"], model=model, robot=robot, dataset=dataset,
                                 tag=tag, episode_id=r["episode_id"])
        if not os.path.exists(os.path.join(pdir, "meta.json")):
            continue
        pred = io.load_prediction(pdir)
        video, npz = io.episode_paths(data_dir=parameters["data_dir"], robot=robot, dataset=dataset, split="test",
                                      episode_id=r["episode_id"])
        res = {}
        if pred["video"] is not None:
            real_v, pred_v = io.read_video(video), io.read_video(pred["video"])
            for mode in PIXEL_MODES[model]:
                res[f"pixel_{mode}"] = pixel.clip_metrics(pred=pred_v, real=real_v, mode=mode, fps=fps)
        true = probe.targets_at(probe_state=np.load(npz)["probe_state"], frame_index=pred["frame_index"], stride=stride)
        res["probe"] = probe.probe_errors(pred=probe.apply_probe(fitted, probe.flatten_features(pred["pred_features"])),
                                          true=true, robot=robot)
        if r["episode_id"] in reals:
            res["probe_floor"] = probe.probe_errors(
                pred=probe.apply_probe(fitted, probe.flatten_features(reals[r["episode_id"]][0])),
                true=probe.targets_at(probe_state=np.load(npz)["probe_state"], frame_index=reals[r["episode_id"]][1],
                                      stride=stride), robot=robot)
        per_episode[r["episode_id"]] = res
        if r.get("group") and r["episode_id"] in reals:
            g = groups.setdefault(r["group"], ({}, {}))
            g[0][r["episode_id"]] = pred["pred_features"]
            g[1][r["episode_id"]] = reals[r["episode_id"]][0][:len(pred["pred_features"])]
    pairs = [p for preds, rs in groups.values() for p in counterfactual.group_pairs(preds=preds, reals=rs)]
    summary = {"n_episodes": len(per_episode), "n_groups": len(groups), "n_pairs": len(pairs),
               "counterfactual_acc": float(np.mean([p[2] for p in pairs])) if pairs else None,
               "probe_train_set": calib or "nominal/train", "probe_epochs": fitted["epochs"]}
    for key in ("probe", "probe_floor"):
        vals = [e[key] for e in per_episode.values() if key in e]
        if vals:
            summary[key] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    for mode in PIXEL_MODES[model]:
        vals = [e[f"pixel_{mode}"]["mean"] for e in per_episode.values() if f"pixel_{mode}" in e]
        if vals:
            summary[f"pixel_{mode}"] = {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
    return {"summary": summary, "episodes": per_episode, "pairs": pairs}


@click.command()
@click.option("--model", required=True, type=click.Choice(sorted(PIXEL_MODES)))
@click.option("--robot", required=True)
@click.option("--tag", required=True, help="checkpoint tag used by the model's prediction entry")
@click.option("--datasets", required=True, help="comma-separated: nominal-test,shift-<s>-L<l>,...")
def main(model, robot, tag, datasets):
    parameters = load_parameters()
    get_robot(robot=robot, parameters=parameters)
    fps = {"franka_libero": 10, "gr1_robocasa": 10, "widowx_bridge": 5, "google_fractal": 3}[robot]
    out_dir = os.path.join(parameters["results_dir"], "panel", model, robot, tag)
    os.makedirs(out_dir, exist_ok=True)
    for dataset in datasets.split(","):
        out = os.path.join(out_dir, f"{dataset}.json")
        if os.path.exists(out):
            continue
        try:
            result = evaluate_dataset(parameters=parameters, model=model, robot=robot, tag=tag, dataset=dataset, fps=fps)
        except FileNotFoundError as e:
            log_warn(f"{model}/{robot}/{dataset}: missing inputs ({e}); skipped", parameters=parameters)
            continue
        with open(out + ".tmp", "w") as f:
            json.dump(result, f)
        os.replace(out + ".tmp", out)
        log_info(f"{model}/{robot}/{dataset}: {json.dumps(result['summary'])}", parameters=parameters)


if __name__ == "__main__":
    main()
