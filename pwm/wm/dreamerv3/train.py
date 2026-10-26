"""Train DreamerV3's world model on one robot's recorded data, until the selection metric stops improving.

    python -m pwm.wm.dreamerv3.train --robot franka_libero --run_name dv3-franka_libero-nominal \
        --ckpt_dir <checkpoints>/dv3-franka_libero-nominal [--dataset nominal] [--override key=value ...]

Settings: configs/world_models.yaml ``dreamerv3``, then ``--override`` (dotted keys for nesting; smoke tests).

Every ``eval_every`` steps, with all randomness fixed:
- the selection number ``val/openloop_mse64`` (pwm/wm/dreamerv3/evaluate.py: 4.8 s open-loop from 1 real frame, 64 px,
  fixed val episodes), which decides the best checkpoint and the stop;
- the base training objective on fixed val windows, split into image / success (reward head) / continue NLL and the KL
  term (``val/base_loss``, ``val/image_loss``, ``val/reward_loss``, ``val/kl_term``, ...).
Stops after ``patience`` evaluations without a relative improvement of ``min_delta`` (or at ``max_steps``), then
reports native-resolution open-loop MSE / LPIPS and action sensitivity for the best checkpoint, writes STOPPED.json and
exports the best weights to <model_dir>/dreamerv3/<run_name>/.

Restart-safe: resumes from the newest readable checkpoint in ``--ckpt_dir`` (every ``save_minutes`` and after every
evaluation; temporary name then os.replace; the last 3 kept plus ``best.pt``). A run that finds STOPPED.json exits.
A small summary (settings, history, best, GPU-hours) is kept in <storage_dir>/runs/dreamerv3/<robot>/<run_name>.json,
readable from the dev box (the checkpoints are on the cluster's /checkpoints volume).
"""
import glob
import json
import os
import shutil
import time

import click
import numpy as np
import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from pwm.robots import get_robot
from pwm.utils import load_parameters, log_info, log_warn, write_meta
from pwm.wm.dreamerv3.data import Replay, action_stats, build_cache, read_index, stack_windows
from pwm.wm.dreamerv3.evaluate import fixed_episodes, openloop_mse64, report
from pwm.wm.dreamerv3.model import (build_world_model, load_checkpoint, make_config, normalize_actions,
                                    save_checkpoint, world_model_loss)
from pwm.wm.dreamerv3.telemetry import Telemetry


class WindowStream(torch.utils.data.IterableDataset):
    """Endless random training windows; each DataLoader worker samples with its own seed."""

    def __init__(self, *, replay: Replay, batch: int, length: int, stats: dict, seed: int):
        self.replay, self.batch, self.length, self.stats, self.seed = replay, batch, length, stats, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            yield normalize_actions(self.replay.sample(rng=rng, batch=self.batch, length=self.length), self.stats)


class FixedRNG:
    """Evaluate with fixed seeds without disturbing the training RNG streams."""

    def __enter__(self):
        self.cpu, self.cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.manual_seed(0)

    def __exit__(self, *exc):
        torch.set_rng_state(self.cpu)
        torch.cuda.set_rng_state(self.cuda)


def validation_objective(*, wm, replay: Replay, starts: list[int], length: int, batch: int, stats: dict) -> dict:
    """The base training objective and its parts on fixed windows."""
    parts, n = {}, 0
    for i in range(0, len(starts), batch):
        windows = [replay.window(s, length) for s in starts[i:i + batch]]
        _, metrics = world_model_loss(wm, normalize_actions(stack_windows(windows), stats))
        for name, v in metrics.items():
            parts[name] = parts.get(name, 0.0) + v * len(windows)
        n += len(windows)
    return {f"val/{k}": v / n for k, v in parts.items()}


def resolve_settings(*, parameters: dict, override: tuple[str, ...]) -> dict:
    """configs/world_models.yaml ``dreamerv3``, then ``--override`` items."""
    s = yaml.safe_load(yaml.safe_dump(parameters["dreamerv3"]))  # deep copy: overrides must not touch parameters
    for item in override:
        key, value = item.split("=", 1)
        node = s
        *path, leaf = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return s


def summary_path(*, parameters: dict, robot: str, run_name: str) -> str:
    d = os.path.join(parameters["storage_dir"], "runs", "dreamerv3", robot)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, run_name + ".json")


def write_json(path: str, obj) -> None:
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


@click.command()
@click.option("--robot", required=True)
@click.option("--run_name", required=True, help="Unique per experiment (also the W&B run id)")
@click.option("--ckpt_dir", required=True, help="Durable checkpoint directory (absolute path)")
@click.option("--dataset", default="nominal", help="Dataset under <data_dir>/<robot>/")
@click.option("--override", multiple=True, help="key=value (dotted keys for nesting; YAML values)")
def main(robot, run_name, ckpt_dir, dataset, override):
    parameters = load_parameters()
    get_robot(robot=robot, parameters=parameters)  # validates the robot name
    s = resolve_settings(parameters=parameters, override=override)
    os.makedirs(ckpt_dir, exist_ok=True)
    stopped = os.path.join(ckpt_dir, "STOPPED.json")
    if os.path.exists(stopped):
        log_info(f"{run_name} already stopped: {open(stopped).read()}", parameters=parameters)
        return
    dataset_dir = os.path.join(parameters["data_dir"], robot, dataset)
    if os.path.exists(os.path.join(dataset_dir, ".hold_dreamerv3")):  # data maintenance in progress: no cache build
        raise SystemExit(f"{dataset_dir} is on hold (.hold_dreamerv3); not starting {run_name}")
    workers = int(os.environ.get("OMP_NUM_THREADS", "8"))
    log = lambda m: log_info(m, parameters=parameters)
    res = s["image_size"]
    caches = {split: build_cache(dataset_dir=dataset_dir, split=split, size=res, workers=workers, log=log)
              for split in ("train", "val")}
    train, val = Replay(cache_split_dir=caches["train"]), Replay(cache_split_dir=caches["val"])
    val64 = val if res == 64 else Replay(cache_split_dir=build_cache(dataset_dir=dataset_dir, split="val", size=64,
                                                                     workers=workers, log=log))
    fps = {r["episode_id"]: float(r.get("fps", 10)) for r in read_index(split_dir=os.path.join(dataset_dir, "val"))}
    sel_episodes = fixed_episodes(episodes=val.episodes, count=s["openloop_episodes"])
    rep_episodes = fixed_episodes(episodes=val.episodes, count=s["report_episodes"])
    log(f"train {len(train.episodes)} episodes / {train.n} frames, val {len(val.episodes)} / {val.n}; "
        f"selection on {len(sel_episodes)} val episodes")

    stats_path = os.path.join(ckpt_dir, "action_stats.json")  # fixed at the first start, reused on every resume
    if not os.path.exists(stats_path):
        write_json(stats_path, action_stats(replay=train))
    with open(stats_path) as f:
        stats = json.load(f)

    torch.manual_seed(s["seed"])
    config = make_config(settings=s, action_dim=train.action_dim)
    wm = build_world_model(config=config, image_size=res)
    n_params = sum(p.numel() for p in wm.parameters())
    st = {"step": 0, "best": float("inf"), "best_step": 0, "history": [], "bad": 0, "wall_s": 0.0}
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")), reverse=True):
        try:
            state = load_checkpoint(path=path, wm=wm)
        except Exception as e:  # truncated or unreadable: try the next older one
            log_warn(f"could not load {path}: {e}", parameters=parameters)
            continue
        st = {k: state.get(k, v) for k, v in st.items()}
        log(f"resumed from {path} (step {st['step']}, best {st['best']:.5f})")
        break

    loader = torch.utils.data.DataLoader(
        WindowStream(replay=train, batch=s["batch_size"], length=s["batch_length"], stats=stats,
                     seed=s["seed"] * 1000003 + st["step"]),
        batch_size=None, num_workers=s["loader_workers"],
        **({"prefetch_factor": 4, "persistent_workers": True, "multiprocessing_context": "fork"}  # fork: workers
           if s["loader_workers"] else {}))                                                        # share the replay
    batches = iter(loader)
    first = next(batches)  # workers fork before any other thread (W&B) starts, avoiding fork-after-threads deadlocks
    val_starts = val.fixed_windows(count=s["val_windows"], length=s["batch_length"], seed=0)
    import lpips
    lpips_net = lpips.LPIPS(net="alex", verbose=False).to(config.device).eval()

    meta = {"robot": robot, "dataset": dataset, "run_name": run_name, "settings": s,
            "n_params": n_params, "action_dim": train.action_dim, "train_frames": train.n, "val_frames": val.n,
            "selection": "val/openloop_mse64"}
    tel = Telemetry(project=parameters["WANDB_PROJECT"], run_id=run_name, total_steps=s["max_steps"], config=meta)
    log(f"DreamerV3 world model: {n_params / 1e6:.1f}M params, actions {train.action_dim}-d, {res}px")
    summary_file = summary_path(parameters=parameters, robot=robot, run_name=run_name)

    def checkpoint(name: str) -> None:
        save_checkpoint(path=os.path.join(ckpt_dir, name), wm=wm, step=st["step"], extra={**st, "meta": meta})

    def rotate() -> None:
        for old in sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))[:-3]:
            os.remove(old)

    def summarize(**extra) -> None:
        if not s.get("export", True):  # throwaway runs (benchmarks) leave no summary
            return
        write_json(summary_file, {**meta, **{k: st[k] for k in ("step", "best", "best_step", "history")},
                                  "gpu_h": st["wall_s"] / 3600.0, **extra})

    flops_per_step, t_save, t_last, reason = None, time.time(), time.time(), None
    while reason is None:
        batch = first if first is not None else next(batches)
        first = None
        if flops_per_step is None:
            with FlopCounterMode(display=False) as fc:
                loss, metrics = world_model_loss(wm, batch)
                wm._model_opt(loss, wm.parameters(), retain_graph=False)
            flops_per_step = fc.get_total_flops()
        else:
            loss, metrics = world_model_loss(wm, batch)
            wm._model_opt(loss, wm.parameters(), retain_graph=False)
        st["step"] += 1
        now = time.time()
        st["wall_s"] += now - t_last
        tel.active_step, tel.step_time_s = st["step"], now - t_last
        tel.effective_mfu = flops_per_step / max(now - t_last, 1e-6) / s["peak_flops"]
        t_last = now
        if st["step"] % s["log_every"] == 0:
            tel.log({"train/loss": float(loss), **{f"train/{k}": v for k, v in metrics.items()}, "train/lr": float(s["model_lr"]),
                     "perf/step_time_s": tel.step_time_s, "perf/effective_mfu": tel.effective_mfu}, st["step"])
            print(f"step {st['step']} loss {float(loss):.4f} base {metrics['base_loss']:.4f} "
                  f"{tel.step_time_s:.3f}s/step mfu {tel.effective_mfu:.4f}", flush=True)
        if st["step"] % s["eval_every"] == 0:
            t0 = time.time()
            wm.eval()
            with torch.no_grad(), FixedRNG():
                v = validation_objective(wm=wm, replay=val, starts=val_starts, length=s["batch_length"],
                                         batch=s["batch_size"], stats=stats)
                v["val/openloop_mse64"] = openloop_mse64(wm=wm, stats=stats, replay=val, replay64=val64,
                                                         episodes=sel_episodes, fps=fps, horizon_s=s["horizon_s"],
                                                         context=s["context_frames"])
            wm.train()
            v["val/loss"] = v["val/openloop_mse64"]  # the telemetry contract's val/loss = the selection number
            st["history"].append({"step": st["step"], **v})
            tel.log(v, st["step"])
            sel = v["val/openloop_mse64"]
            if sel < st["best"] - s["min_delta"] * abs(st["best"] if np.isfinite(st["best"]) else 1.0):
                st["best"], st["best_step"], st["bad"] = sel, st["step"], 0
                checkpoint("best.pt")
            else:
                st["bad"] += 1
            log(f"step {st['step']} openloop_mse64 {sel:.5f} base {v['val/base_loss']:.3f} (best {st['best']:.5f} "
                f"@ {st['best_step']}, {st['bad']}/{s['patience']} without improvement, eval {time.time() - t0:.0f}s)")
            if st["bad"] >= s["patience"]:
                reason = "selection metric plateau"
            checkpoint(f"step_{st['step']:08d}.pt")
            rotate()
            summarize()
            st["wall_s"] += time.time() - t0  # evaluation time is GPU time too
            t_save = t_last = time.time()
        if st["step"] >= s["max_steps"] and reason is None:
            reason = "max_steps"
        if reason is None and time.time() - t_save > s["save_minutes"] * 60:
            checkpoint(f"step_{st['step']:08d}.pt")
            rotate()
            t_save = time.time()

    checkpoint(f"step_{st['step']:08d}.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    final = {}
    if os.path.exists(best_path):  # per-run report columns on the best checkpoint
        t0 = time.time()
        load_checkpoint(path=best_path, wm=wm, with_optimizer=False)
        wm.eval()
        with torch.no_grad(), FixedRNG():
            final = report(wm=wm, stats=stats, replay=val, episodes=rep_episodes, fps=fps, horizon_s=s["horizon_s"],
                           context=s["context_frames"], lpips_net=lpips_net)
        best_eval = next((h for h in st["history"] if h["step"] == st["best_step"]), {})
        final.update({k: best_eval.get(k) for k in ("val/base_loss", "val/image_loss", "val/reward_loss",
                                                     "val/cont_loss", "val/kl_term", "val/kl")})
        st["wall_s"] += time.time() - t0
    summarize(stop_reason=reason, final=final)
    if s.get("export", True):
        export = os.path.join(parameters["model_dir"], "dreamerv3", run_name)
        os.makedirs(export, exist_ok=True)
        if os.path.exists(best_path):
            shutil.copy(best_path, os.path.join(export, "best.pt.tmp"))
            os.replace(os.path.join(export, "best.pt.tmp"), os.path.join(export, "best.pt"))
        shutil.copy(stats_path, os.path.join(export, "action_stats.json"))
        write_meta(export, {**meta, "stop_reason": reason, "stop_step": st["step"], "best": st["best"],
                            "best_step": st["best_step"], "final": final, "history": st["history"]}, parameters)
    write_json(stopped, {"reason": reason, "step": st["step"], "best": st["best"], "best_step": st["best_step"],
                         "final": final})
    for old in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):  # resume states are dead weight once stopped
        os.remove(old)
    tel.close({"stop_reason": reason, "best_selection": st["best"]})
    log(f"{run_name} stopped at step {st['step']} ({reason}); best openloop_mse64 {st['best']:.5f} @ {st['best_step']}")


if __name__ == "__main__":
    main()
