"""Fine-tune V-JEPA 2-AC's predictor on one robot's data until the validation loss stops improving.

Frozen encoder, trainable predictor (as in Meta's DROID post-training); V-JEPA 2-AC's own objective (teacher-forced +
autoregressive L1 in feature space, pwm/wm/vjepa2/model.py:losses). Single GPU or torchrun (DDP).

Checkpoints in --out: `step_<n>.pt` (predictor, optimizer, progress), written every `save_every_minutes` from a background
thread to a temporary name and then renamed, newest `keep_checkpoints` kept; a restarted run resumes from the newest
readable one. `best.pt` holds the predictor with the lowest validation loss. Every `eval_every_steps` the loss on a fixed
set of validation windows is computed; after `patience` evaluations without improvement the run writes `STOPPED.json`
and exits 0 (a restarted job that finds STOPPED.json exits immediately).

Validation is identical for every run, whatever its training settings (VAL_* below, not configurable): the same windows
of the val split (8 model steps from evenly spaced starts, 512 at most), reporting
  val/loss_tf  L1 of one teacher-forced pass (each of frames 1..7 predicted from the real earlier frames)
  val/loss_ar  L1 of a free-running 7-step rollout from the window's first frame (each frame from the model's own
               earlier predictions) -- **the selection number** (best.pt, early stopping): it is what the evaluation
               protocol does (free-running from one context frame).
Settings: configs/world_models.yaml `vjepa2` (the robot's entry merged in), then --override.

    python -m pwm.wm.vjepa2.train --robot franka_libero --dataset nominal --out <dir> [--override key=value ...]
"""
import glob
import json
import os
import socket
import threading
import time

import click
import yaml
import numpy as np
import torch
import torch.distributed as dist

from pwm.utils import load_parameters, log_info
from pwm.wm.vjepa2 import data, model, rollout_eval
from pwm.wm.vjepa2.telemetry import Telemetry

HEARTBEAT_STALE_S = 600  # a run whose heartbeat is younger than this is owned by another live process
VAL_LENGTH = 8           # model steps per validation window
VAL_ROLLOUT_STEPS = 7    # free-running steps of val/loss_ar (the whole window after its first frame)
VAL_MAX_WINDOWS = 512
SELECT = "val/loss_ar"


def learning_rate(cfg, step: int) -> float:
    """Linear warm-up, then constant."""
    return cfg["lr"] * min(1.0, (step + 1) / cfg["warmup_steps"])


def _latest_checkpoint(out: str):
    for path in sorted(glob.glob(os.path.join(out, "step_*.pt")), reverse=True):
        try:
            return torch.load(path, map_location="cpu", weights_only=False), path
        except Exception as e:  # a checkpoint cut off by an eviction: try the previous one
            print(f"[vjepa2] unreadable checkpoint {path}: {e!r}", flush=True)
    return None, None


class Saver:
    """Background checkpoint writes (CPU copy first, then file I/O off the training thread). A failed write is re-raised
    on the training thread at the next save or wait, so the job fails instead of training on without checkpoints."""

    def __init__(self, out: str, keep: int):
        self.out, self.keep, self.thread, self.error = out, keep, None, None

    def save(self, name: str, state: dict, rotate: bool):
        self.wait()

        def write():
            path = os.path.join(self.out, name)
            try:
                torch.save(state, path + ".tmp")
                os.replace(path + ".tmp", path)
            except BaseException as e:
                self.error = e
                if os.path.exists(path + ".tmp"):
                    os.remove(path + ".tmp")
                return
            if rotate:
                for old in sorted(glob.glob(os.path.join(self.out, "step_*.pt")))[: -self.keep]:
                    os.remove(old)

        self.thread = threading.Thread(target=write)
        self.thread.start()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            raise RuntimeError(f"checkpoint write to {self.out} failed") from self.error


def _cpu(obj):
    """Deep CPU copy (tensors keep changing on the GPU while the background thread writes)."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(v) for v in obj)
    return obj


@torch.no_grad()
def validate(*, predictor, encoder, episodes, windows, cfg, rank, world, device):
    """Validation losses over the fixed windows (sharded over ranks); see the module docstring."""
    predictor.eval()
    sums = torch.zeros(3, device=device)  # tf, ar, count
    mine = windows[rank::world]
    bs = cfg["batch_size"]
    for i in range(0, len(mine), bs):
        batch = [episodes[e].window(start=s, length=VAL_LENGTH) for e, s in mine[i : i + bs]]
        frames = torch.stack([data.to_model_input(f) for f, _, _ in batch]).to(device)
        actions = torch.from_numpy(np.stack([a for _, a, _ in batch])).to(device)
        states = torch.from_numpy(np.stack([s for _, _, s in batch])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(encoder, frames)
            tf, ar = model.losses(predictor, h, actions, states, VAL_ROLLOUT_STEPS)
        n = len(batch)
        sums += torch.tensor([tf.item() * n, ar.item() * n, n], device=device)
    if world > 1:
        dist.all_reduce(sums)
    predictor.train()
    tf, ar, n = sums.tolist()
    return {"val/loss_tf": tf / n, "val/loss_ar": ar / n}


@click.command()
@click.option("--robot", required=True)
@click.option("--dataset", default="nominal")
@click.option("--out", required=True, help="Checkpoint directory (absolute)")
@click.option("--max_steps", default=0, type=int, help="Hard cap on steps (0: config max_steps)")
@click.option("--override", multiple=True, help="key=value over the vjepa2 config (YAML value), e.g. eval_every_steps=50")
def main(robot, dataset, out, max_steps, override):
    parameters = load_parameters()
    cfg = data.robot_cfg(parameters=parameters, robot=robot)
    overrides = {}
    for item in override:
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    for key, value in overrides.items():
        assert key in cfg, f"unknown vjepa2 setting {key}"
        cfg[key] = value
    cfg["overrides"] = overrides
    max_steps = max_steps or cfg["max_steps"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl", timeout=__import__("datetime").timedelta(minutes=60))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "STOPPED.json")):
        log_info(f"[vjepa2] {out} already STOPPED; nothing to do", parameters=parameters)
        return
    beat = os.path.join(out, "HEARTBEAT")
    if rank == 0 and os.path.exists(beat) and time.time() - os.path.getmtime(beat) < HEARTBEAT_STALE_S:
        with open(beat) as f:
            owner = f.read().strip()
        if owner != f"{socket.gethostname()}:{os.getpid()}":
            log_info(f"[vjepa2] {out} is being trained by {owner}; exiting", parameters=parameters)
            return
    if rank == 0:  # claim the run before the slow start-up (model build ~3 min)
        with open(beat + ".tmp", "w") as f:
            f.write(f"{socket.gethostname()}:{os.getpid()}")
        os.replace(beat + ".tmp", beat)

    length = cfg["frames_per_clip"]
    train_eps = data.load_episodes(parameters=parameters, robot=robot, dataset=dataset, split="train", min_frames=length)
    val_eps = data.load_episodes(parameters=parameters, robot=robot, dataset=dataset, split="val", min_frames=VAL_LENGTH)
    assert train_eps and val_eps, "no cached train/val episodes: run pwm.wm.vjepa2.data build-cache first"
    windows = data.val_windows(episodes=val_eps, length=VAL_LENGTH, max_windows=VAL_MAX_WINDOWS)
    action_dim, state_dim = train_eps[0].actions.shape[1], train_eps[0].state.shape[1]

    encoder, predictor = model.build(parameters=parameters, action_dim=action_dim, state_dim=state_dim, device=device,
                                     predictor_init=cfg["predictor_init"], predictor_depth=cfg["predictor_depth"])
    predictor.use_activation_checkpointing = cfg["activation_checkpointing"]
    predictor.extrinsics_encoder.requires_grad_(False)  # unused without camera extrinsics
    params = [p for p in predictor.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.999))

    step, best, bad, history, gpu_s = 0, float("inf"), 0, [], 0.0
    ckpt, ckpt_path = _latest_checkpoint(out)
    if ckpt is not None:
        predictor.load_state_dict(ckpt["predictor"])
        opt.load_state_dict(ckpt["optimizer"])
        step, best, bad, history, gpu_s = ckpt["step"], ckpt["best"], ckpt["bad"], ckpt["history"], ckpt["gpu_s"]
        log_info(f"[vjepa2] resumed from {ckpt_path} (step {step}, best val {best:.5f})", parameters=parameters)
    del ckpt
    gpu_s0, t_process = gpu_s, time.time()
    ddp = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[local]) if world > 1 else predictor

    loader = torch.utils.data.DataLoader(
        data.Windows(episodes=train_eps, length=length, seed=cfg["seed"], rank=rank, resume_step=step),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True,
        prefetch_factor=4, multiprocessing_context="forkserver")
    batches = iter(loader)
    first = next(batches)  # before telemetry threads start (fork-after-threads deadlock seen with W&B)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    tok = model.TOKENS_PER_FRAME + 2
    flops = cfg["batch_size"] * world * (2 * n_enc * length * model.TOKENS_PER_FRAME + 6 * n_pred * (length + 1) * tok)
    tel = Telemetry("pwm-vjepa2", max_steps, flops, world,
                    config={**{k: v for k, v in cfg.items() if not isinstance(v, (dict, list))},
                            "robot": robot, "dataset": dataset, "world": world}) \
        if rank == 0 else None
    saver = Saver(out, cfg["keep_checkpoints"]) if rank == 0 else None
    if rank == 0:
        log_info(f"[vjepa2] {robot}/{dataset}: {len(train_eps)} train / {len(val_eps)} val episodes, "
                 f"{len(windows)} val windows, action {action_dim}-D, state {state_dim}-D, world {world}, "
                 f"bs/gpu {cfg['batch_size']}, predictor {n_pred / 1e6:.0f}M params", parameters=parameters)

    t_save, t_last, stop, t_beat = time.time(), time.time(), False, 0.0
    while not stop:
        frames, actions, states = first if first is not None else next(batches)
        first = None
        frames, actions, states = (x.to(device, non_blocking=True) for x in (frames, actions, states))
        lr = learning_rate(cfg, step)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(encoder, frames)
            jl, sl = model.losses(ddp, h, actions, states, cfg["loss_rollout_steps"])
            loss = jl + sl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        step += 1
        now = time.time()
        gpu_s = gpu_s0 + (now - t_process) * world  # GPU time held, incl. start-up and evaluations
        if rank == 0:
            tel.update(step, now - t_last, {"train/loss": loss.item(), "train/loss_tf": jl.item(),
                                            "train/loss_ar": sl.item(), "train/grad_norm": gn.item(), "train/lr": lr})
            if step % 20 == 0 or step <= 3:
                print(f"step {step} loss {loss.item():.4f} (tf {jl.item():.4f} ar {sl.item():.4f}) gn {gn.item():.3f} "
                      f"lr {lr:.2e} {now - t_last:.2f}s/step mfu {tel.mfu:.3f} "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.0f}GiB", flush=True)
        t_last = now
        if rank == 0 and now - t_beat > 60:
            with open(beat + ".tmp", "w") as f:
                f.write(f"{socket.gethostname()}:{os.getpid()}")
            os.replace(beat + ".tmp", beat)
            t_beat = now

        flags = torch.zeros(2, device=device)  # [stop, save]
        if step % cfg["eval_every_steps"] == 0:
            v = validate(predictor=predictor, encoder=encoder, episodes=val_eps, windows=windows, cfg=cfg, rank=rank,
                         world=world, device=device)
            if rank == 0:
                improved = v[SELECT] < best * (1 - cfg["min_rel_improvement"])
                if improved:
                    best, bad = v[SELECT], 0
                    saver.save("best.pt", {"predictor": _cpu(predictor.state_dict()), "step": step, "val": v,
                                           "action_dim": action_dim, "state_dim": state_dim, "robot": robot,
                                           "dataset": dataset, "cfg": cfg, "gpu_s": gpu_s}, rotate=False)
                else:
                    bad += 1
                history.append({"step": step, **v, "best": best, "bad": bad, "gpu_h": gpu_s / 3600})
                with open(os.path.join(out, "progress.json.tmp"), "w") as f:  # readable without torch
                    json.dump({"step": step, "gpu_h": gpu_s / 3600, "history": history}, f)
                os.replace(os.path.join(out, "progress.json.tmp"), os.path.join(out, "progress.json"))
                print(f"[val] step {step} " + " ".join(f"{k} {x:.5f}" for k, x in v.items())
                      + f" best {best:.5f} bad {bad}/{cfg['patience']}", flush=True)
                tel.update(step, time.time() - t_last, v)
                flags[0] = float(bad >= cfg["patience"] or step >= max_steps)
            t_last = time.time()
        elif rank == 0 and step >= max_steps:
            flags[0] = 1.0
        if rank == 0 and (time.time() - t_save > 60 * cfg["save_every_minutes"] or flags[0]):
            flags[1] = 1.0
        if world > 1:
            dist.broadcast(flags, 0)
        stop = bool(flags[0])
        if rank == 0 and flags[1]:
            saver.save(f"step_{step:07d}.pt", {"predictor": _cpu(predictor.state_dict()),
                                               "optimizer": _cpu(opt.state_dict()), "step": step, "best": best, "bad": bad,
                                               "history": history, "action_dim": action_dim, "state_dim": state_dim,
                                               "gpu_s": gpu_s, "cfg": cfg},
                       rotate=True)
            t_save = time.time()

    if rank == 0:
        saver.wait()
        reason = "patience" if bad >= cfg["patience"] else "max_steps"
        with open(os.path.join(out, "STOPPED.json.tmp"), "w") as f:
            json.dump({"reason": reason, "step": step, "select": SELECT, "best_val": best, "gpu_h": gpu_s / 3600,
                       "overrides": overrides, "history": history}, f, indent=1)
        os.replace(os.path.join(out, "STOPPED.json.tmp"), os.path.join(out, "STOPPED.json"))
        for old in glob.glob(os.path.join(out, "step_*.pt")):  # resume states are dead weight once stopped
            os.remove(old)
        log_info(f"[vjepa2] stopped at step {step} ({reason}); best val {best:.5f}", parameters=parameters)
        tel.finish()
        # rollout metrics of best.pt (pwm/wm/vjepa2/rollout_eval.py; the GPU is free now)
        del ddp, opt
        torch.cuda.empty_cache()
        rollout_eval.run(parameters=parameters, robot=robot, dataset=dataset, checkpoint=os.path.join(out, "best.pt"),
                         encoder=encoder, device=device)
        print(f"STOPPED {out}", flush=True)  # job-log marker: cluster/sail/reaper.sh cancels jobs that linger after it
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
