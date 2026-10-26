"""Fine-tune DreamDojo 2B on one robot's episodes, conditioned on native actions (configs/world_models.yaml,
dreamdojo). Run with torch.distributed.run, one process per GPU (scripts/jobs/train_dreamdojo.sh).

- Start: the released 2B pretrained EMA weights (or --init). The DiT (`model.net`) trains in fp32 master weights under
  bf16 autocast; tokenizer, conditioner and the cached "" text embedding are frozen. Scope `full`: all 2.15B DiT
  parameters; scope `lora`: LoRA on the blocks' attention / MLP projections + the action embedders (pwm.wm.dreamdojo.lora).
- Settings: exactly configs/world_models.yaml `dreamdojo.train`. `effective_batch` clips per optimiser step on any
  number of GPUs: micro-batch (<= max_micro_batch) x GPUs x gradient accumulation.
- Data: `<dataset>/train` clips (pwm.wm.dreamdojo.data), actions normalised with train-split stats (run_dir).
- Gradients: one flat fp32 buffer, one all_reduce per optimiser step. AdamW, linear warm-up then constant LR. EMA of the
  trainable parameters on every rank (identical updates), so validation and exports use EMA weights.
- Validation (identical for every run): the fixed `val_clips` clips of `<dataset>/val`, each with its own fixed seed
  (fixed noise and noise level), EMA weights, every `val_every_clips` training clips. Stop when the val loss has not
  improved for `val_patience` consecutive evaluations (or after `max_epochs`); <run_dir>/STOPPED records it and
  best_ema_bf16.pt holds the best weights. progress.json is rewritten after every evaluation.
- `seed` changes only the training-data order; validation is fixed regardless.
- Small status files (run_info.json, progress.json, STOPPED) are also mirrored to --status_dir (default
  <results_dir>/runs/dreamdojo/<run_dir name>/), readable from the dev box where /checkpoints is not mounted.
- Restart-safe: full checkpoints every `save_minutes` and after every evaluation (rank 0: CPU copy then background
  write, tmp + os.replace, last `keep_checkpoints` kept); the newest readable one is resumed. Stop / save decisions are
  made on rank 0 and broadcast.
"""
import datetime
import glob
import json
import os
import threading
import time

import click
import torch
import torch.distributed as dist

from pwm.utils import load_parameters
from pwm.wm.dreamdojo import data as D
from pwm.wm.dreamdojo import lora as L
from pwm.wm.dreamdojo import model as M
from pwm.wm.dreamdojo.assets import init_weights_path
from pwm.wm.dreamdojo.telemetry import Telemetry

def say(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def checkpoints(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "step_*.pt")))


def clip_epoch_size(*, episodes: list[dict], robot_cfg: dict, n: int) -> int:
    s = robot_cfg["commands_per_frame"]
    return int(sum(max(0, e["steps"] // s + 1 - n + 1) for e in episodes))


def micro_and_accum(*, effective_batch: int, world: int, max_micro: int) -> tuple[int, int]:
    per_gpu = effective_batch // world
    if per_gpu * world != effective_batch:
        raise ValueError(f"effective batch {effective_batch} is not divisible by {world} GPUs")
    micro = max(m for m in range(1, max_micro + 1) if per_gpu % m == 0)
    return micro, per_gpu // micro


def write_json(path: str, obj) -> None:
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


@click.command()
@click.option("--robot", required=True)
@click.option("--dataset", required=True, help="Dataset root (<storage>/data/<robot>/<dataset>), with train/ and val/")
@click.option("--run_dir", required=True, help="Checkpoints and markers (absolute path, under /checkpoints)")
@click.option("--init", default="none", help="Inference-format weights to start from; none = released 2B pretrain")
@click.option("--max_steps", default=0, type=int, help="Optimiser steps cap (0 = until the stopping rule)")
@click.option("--stop_at", default=0.0, type=float, help="Unix time to save and exit (0 = never)")
@click.option("--val_every_clips", default=0, type=int, help="Override val_every_clips (0 = config); smoke tests")
@click.option("--save_every", default=0, type=int, help="Also save every this many optimiser steps (0 = time-based)")
@click.option("--status_dir", default="none", help="Mirror of the run's small status files (run_info, progress, "
              "STOPPED); none = <results_dir>/runs/dreamdojo/<run_dir name>")
def main(robot, dataset, run_dir, init, max_steps, stop_at, val_every_clips, save_every, status_dir):
    parameters = load_parameters()
    cfg = parameters["dreamdojo"]
    tcfg = dict(cfg["train"])
    if val_every_clips:
        tcfg["val_every_clips"] = val_every_clips
    rcfg = cfg["robots"][robot]
    n, size = cfg["clip_frames"], tuple(cfg["resolution"])
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    os.makedirs(run_dir, exist_ok=True)
    if status_dir == "none":
        status_dir = os.path.join(parameters["results_dir"], "runs", "dreamdojo", os.path.basename(run_dir.rstrip("/")))
    os.makedirs(status_dir, exist_ok=True)

    def status(name: str, obj) -> None:
        """Status files go to the run directory and to its mirror (readable where checkpoints are not mounted)."""
        write_json(os.path.join(run_dir, name), obj)
        write_json(os.path.join(status_dir, name), obj)

    if os.path.exists(os.path.join(run_dir, "STOPPED")):
        say(f"{run_dir} already STOPPED: nothing to do")
        dist.destroy_process_group()
        return
    micro, accum = micro_and_accum(effective_batch=tcfg["effective_batch"], world=world,
                                   max_micro=tcfg["max_micro_batch"])
    if rank == 0:
        status("run_info.json", {"robot": robot, "dataset": dataset, "settings": tcfg, "init": init, "run_dir": run_dir})

    # ---------------------------------------------------------------- data and action statistics
    train_eps = D.read_episodes(split_dirs=[os.path.join(dataset, "train")])
    val_eps = D.read_episodes(split_dirs=[os.path.join(dataset, "val")])
    stats = os.path.join(run_dir, "action_stats.json")
    if rank == 0 and not os.path.exists(stats):
        sample = sorted(train_eps, key=lambda e: e["episode_id"])
        sample = sample[:: max(1, len(sample) // 3000)]
        D.ActionNormaliser.fit(episodes=sample, robot_cfg=rcfg).save(stats)
    dist.barrier()
    norm = D.ActionNormaliser.load(path=stats, robot_cfg=rcfg)
    epoch_clips = clip_epoch_size(episodes=train_eps, robot_cfg=rcfg, n=n)
    say(f"{robot}: {len(train_eps)} train / {len(val_eps)} val episodes, {epoch_clips} train clip starts; "
        f"{world} GPUs x {micro} clips x {accum} accumulation = {tcfg['effective_batch']}")

    # ---------------------------------------------------------------- model, optimiser, EMA
    model = M.load(parameters=parameters, weights=init_weights_path(parameters) if init == "none" else init)
    for p in model.parameters():
        p.requires_grad_(False)
    net = model.net
    net.float().train()
    if tcfg["scope"] == "lora":
        params = L.apply(net=net, rank=tcfg["lora_rank"], alpha=tcfg["lora_alpha"])
    elif tcfg["scope"] == "full":
        params = list(net.parameters())
        for p in params:
            p.requires_grad_(True)
    else:
        raise click.BadParameter(f"unknown scope {tcfg['scope']}")
    n_params = sum(p.numel() for p in params)
    n_net = sum(p.numel() for p in net.parameters())
    flat = torch.zeros(n_params, device="cuda", dtype=torch.float32)
    off = 0
    for p in params:  # grads live in one flat buffer -> one all_reduce per optimiser step
        p.grad = flat[off:off + p.numel()].view_as(p)
        off += p.numel()
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], betas=(0.9, 0.99), weight_decay=tcfg["weight_decay"], eps=1e-8)
    ema = [p.detach().clone() for p in params]
    st = {"step": 0, "clips": 0, "best_val": None, "best_step": None, "best_clips": None, "bad_evals": 0,
          "val_history": [], "gpu_seconds": 0.0, "wall_seconds": 0.0, "next_val_clips": tcfg["val_every_clips"]}

    for f in reversed(checkpoints(run_dir)):  # resume from the newest readable checkpoint
        try:
            c = torch.load(f, map_location="cpu", weights_only=False)
            if len(c["net"]) != len(params):
                raise RuntimeError(f"{len(c['net'])} saved tensors vs {len(params)} trainable (scope changed?)")
            for p, v in zip(params, c["net"]):
                p.data.copy_(v)
            for e, v in zip(ema, c["ema"]):
                e.copy_(v)
            opt.load_state_dict(c["opt"])
            for p in params:  # load_state_dict must not break the flat-grad views
                p.grad = p.grad
            st.update(c["state"])
            say(f"resumed from {f} (step {st['step']}, clips {st['clips']})")
            break
        except Exception as e:
            say(f"unreadable checkpoint {f}: {e!r}")
    say(f"trainable {n_params / 1e9:.3f}B of {n_net / 1e9:.3f}B params ({tcfg['scope']}), lr {tcfg['lr']}")

    # ---------------------------------------------------------------- loaders (before any W&B thread exists)
    loader = torch.utils.data.DataLoader(
        D.TrainClips(episodes=train_eps, normaliser=norm, robot_cfg=rcfg, n=n, size=size,
                     seed=1 + 1000 * tcfg["seed"]),  # seed 0 = the original data order
        batch_size=micro, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True,
        worker_init_fn=D.worker_init, multiprocessing_context="forkserver")
    it = iter(loader)
    first = next(it)
    print(f"rank {rank}: first batch at {time.strftime('%H:%M:%S')}", flush=True)
    val_all = D.fixed_val_clips(episodes=val_eps, normaliser=norm, robot_cfg=rcfg, n=n, size=size,
                                count=tcfg["val_clips"])
    val_mine = [(i, c) for i, c in enumerate(val_all) if i % world == rank]
    say(f"{len(val_all)} fixed validation clips, every {tcfg['val_every_clips']} training clips")

    total = max_steps or int(tcfg["max_epochs"] * epoch_clips / tcfg["effective_batch"])
    tokens = 4 * (size[0] // 16) * (size[1] // 16)  # latent frames x patches (VAE /8, patch 2)
    # FLOPs per optimiser step: forward + backward through all DiT weights (LoRA still needs input grads: ~ the same)
    tel = Telemetry(project="pwm-dreamdojo", total_steps=total, flops_per_step=6 * n_net * tokens *
                    tcfg["effective_batch"], n_gpus=world,
                    config=dict(robot=robot, dataset=dataset, world=world, micro=micro,
                                accum=accum, **tcfg)) if rank == 0 else None
    saver = {"thread": None, "error": None}

    def wait_saver():
        """Join the background write; a failed one fails the job rather than training on without checkpoints."""
        if saver["thread"] is not None:
            saver["thread"].join()
            saver["thread"] = None
        if saver["error"] is not None:
            raise RuntimeError(f"checkpoint write to {run_dir} failed") from saver["error"]

    def export(values):
        if tcfg["scope"] == "lora":
            return L.export_state_dict(net=net, params=params, values=values)
        return M.export_inference_weights(net=net, ema_params=values)

    def save(final: bool = False):
        """rank 0: CPU copy now, file writes in a background thread."""
        wait_saver()
        state ={"net": [p.detach().cpu() for p in params], "ema": [e.detach().cpu() for e in ema],
                 "opt": {"state": {k: {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                                   for k, v in opt.state_dict()["state"].items()},
                         "param_groups": opt.state_dict()["param_groups"]},
                 "state": dict(st)}
        weights = export(state["ema"])
        step_now = st["step"]

        def write():
            try:
                p = os.path.join(run_dir, f"step_{step_now:07d}.pt")
                torch.save(state, p + ".tmp")
                os.replace(p + ".tmp", p)
                q = os.path.join(run_dir, "ema_bf16_latest.pt")
                torch.save(weights, q + ".tmp")
                os.replace(q + ".tmp", q)
            except BaseException as e:
                saver["error"] = e
                for t in glob.glob(os.path.join(run_dir, "*.tmp")):
                    os.remove(t)
                return
            for old in checkpoints(run_dir)[:-tcfg["keep_checkpoints"]]:
                os.remove(old)
            print(f"saved {p}", flush=True)

        saver["thread"] = threading.Thread(target=write)
        saver["thread"].start()
        if final:
            wait_saver()

    def export_best():
        """Best EMA weights (inference format)."""
        q = os.path.join(run_dir, "best_ema_bf16.pt")
        torch.save(export([e.detach().cpu() for e in ema]), q + ".tmp")
        os.replace(q + ".tmp", q)

    def progress(done: bool, reason: str | None = None) -> dict:
        return {"step": st["step"], "clips": st["clips"],
                "epochs": st["clips"] / max(epoch_clips, 1), "best_val": st["best_val"], "best_step": st["best_step"],
                "best_clips": st["best_clips"], "bad_evals": st["bad_evals"], "val_history": st["val_history"],
                "gpu_hours": st["gpu_seconds"] / 3600, "wall_hours": st["wall_seconds"] / 3600, "world": world,
                "settings": tcfg,
                "done": done, "reason": reason, "best_weights": os.path.join(run_dir, "best_ema_bf16.pt"),
                "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}

    @torch.no_grad()
    def validate() -> float:
        backup = [p.detach().clone() for p in params]
        for p, e in zip(params, ema):
            p.data.copy_(e)
        net.eval()
        total_loss = torch.zeros(2, device="cuda")
        for i, (frames, acts) in val_mine:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(10_000 + i)
                torch.cuda.manual_seed(10_000 + i)
                b = M.batch(frames=torch.from_numpy(frames)[None].cuda(), actions=torch.from_numpy(acts)[None].cuda())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model.forward(b)
            total_loss += torch.stack([loss.detach().float(), torch.ones((), device="cuda")])
        dist.all_reduce(total_loss)
        for p, v in zip(params, backup):
            p.data.copy_(v)
        net.train()
        return float(total_loss[0] / total_loss[1].clamp(min=1))

    # ---------------------------------------------------------------- loop (one iteration = one optimiser step)
    t_save, t_last, pending = time.time(), time.time(), first
    while True:
        lr = tcfg["lr"] * min(1.0, (st["step"] + 1) / tcfg["warmup_steps"])
        for g in opt.param_groups:
            g["lr"] = lr
        flat.zero_()
        loss_sum = torch.zeros((), device="cuda")
        for _ in range(accum):
            frames, acts = pending
            b = M.batch(frames=frames.cuda(non_blocking=True), actions=acts.cuda(non_blocking=True))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model.forward(b)
            (loss / accum).backward()
            loss_sum += loss.detach().float() / accum
            pending = next(it)
        dist.all_reduce(flat)
        flat.div_(world)
        gn = torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        opt.step()
        torch._foreach_lerp_(ema, [p.detach() for p in params], 1 - tcfg["ema"])
        st["step"] += 1
        st["clips"] += tcfg["effective_batch"]
        dist.all_reduce(loss_sum)
        now = time.time()
        dt = now - t_last
        st["gpu_seconds"] += dt * world
        st["wall_seconds"] += dt
        if rank == 0:
            tel.update(step=st["step"], step_time=dt,
                       logs={"train/loss": loss_sum.item() / world, "train/grad_norm": gn.item(), "train/lr": lr,
                             "train/epoch": st["clips"] / max(epoch_clips, 1), "train/clips": st["clips"]})
            if st["step"] % 20 == 0 or st["step"] < 5:
                print(f"step {st['step']} clips {st['clips']} loss {loss_sum.item() / world:.4f} gn {gn.item():.3f} "
                      f"lr {lr:.2e} {dt:.2f}s/step mfu {tel.mfu:.3f} "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.0f}GiB", flush=True)
        t_last = now

        # rank 0 decides: [0] stop (rule met), [1] save, [2] save + exit (time budget)
        flag = torch.zeros(3, device="cuda")
        if st["clips"] >= st["next_val_clips"]:
            st["next_val_clips"] += tcfg["val_every_clips"]
            v = validate()
            if rank == 0:
                if st["best_val"] is None or v < st["best_val"]:
                    st["best_val"], st["best_step"], st["best_clips"], st["bad_evals"] = v, st["step"], st["clips"], 0
                    export_best()
                else:
                    st["bad_evals"] += 1
                st["val_history"].append([st["step"], st["clips"], v])
                tel.log(step=st["step"], logs={"val/loss": v, "val/best": st["best_val"]})
                print(f"val step {st['step']} clips {st['clips']} loss {v:.5f} best {st['best_val']:.5f} "
                      f"(step {st['best_step']}) bad {st['bad_evals']}/{tcfg['val_patience']}", flush=True)
                status("progress.json", progress(done=False))
                if st["bad_evals"] >= tcfg["val_patience"]:
                    flag[0] = 1
            flag[1] = 1
            t_last = time.time()  # validation time is not training step time
        if rank == 0:
            if st["step"] >= total:
                flag[0] = 1
            if (stop_at and now > stop_at) or now - t_save > tcfg["save_minutes"] * 60 or \
                    (save_every and st["step"] % save_every == 0):
                flag[1] = 1
            if stop_at and now > stop_at:
                flag[2] = 1
        dist.broadcast(flag, 0)
        if flag[0] or flag[1]:
            if rank == 0:
                save(final=bool(flag[0] or flag[2]))
            t_save = time.time()
        if flag[0]:
            if rank == 0:
                reason = "val loss plateau" if st["bad_evals"] >= tcfg["val_patience"] else "step/epoch cap"
                status("progress.json", progress(done=True, reason=reason))
                status("STOPPED", progress(done=True, reason=reason))
                # resume states are dead weight once stopped; best_ema_bf16.pt stays
                for old in checkpoints(run_dir) + [os.path.join(run_dir, "ema_bf16_latest.pt")]:
                    if os.path.exists(old):
                        os.remove(old)
                print(f"STOPPED: {reason}; best val {st['best_val']} at step {st['best_step']}", flush=True)
            break
        if flag[2]:
            break
    if rank == 0:
        tel.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
