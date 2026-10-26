"""Restart-safe data-generation worker. Run any number of these, on any nodes, inside the robot's simulator
environment; each takes the next unclaimed unit, runs it, and writes its episodes. A unit is claimed by creating
claims/<unit_id> (O_EXCL) and finished by claims/<unit_id>.done; a claim whose file has not been touched for
`stale_min` minutes (evicted worker) is taken over, skipping episodes already written. Nominal datasets stop taking
units of a source once that source's hour budget (plan.yaml) is met.

    python -m pwm.data.worker --robot widowx_bridge --kind nominal
    python -m pwm.data.worker --robot widowx_bridge --kind test                       # nominal counterfactual groups
    python -m pwm.data.worker --robot widowx_bridge --kind test --shift latency --level 3
    python -m pwm.data.worker --robot widowx_bridge --kind calib --shift latency --level 3
"""
import json
import os
import socket
import time
import zlib

import click
import numpy as np

from pwm.data.perturb import perturb
from pwm.data.units import dataset_dir, plan_groups, plan_nominal, write_plan
from pwm.robots import get_robot
from pwm.sim.backends import make_backend
from pwm.sim.backends.base import run_episode
from pwm.sim.episode import RolloutDir
from pwm.utils import load_parameters, log_info, log_warn


def dataset_name(*, kind, shift, level):
    if kind == "nominal":
        return "nominal"
    if shift is None:
        return "nominal-test" if kind == "test" else "nominal-calib"
    return f"{'shift' if kind == 'test' else 'calib'}-{shift}-L{level}"


class Claims:
    def __init__(self, *, root, stale_min):
        self.dir = os.path.join(root, "claims")
        os.makedirs(self.dir, exist_ok=True)
        self.stale_s = stale_min * 60

    def done(self, uid):
        return os.path.exists(os.path.join(self.dir, uid + ".done"))

    def claim(self, uid):
        path = os.path.join(self.dir, uid)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time()}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(path) > self.stale_s:
                    os.utime(path)
                    return True
            except OSError:
                pass
            return False

    def touch(self, uid):
        try:
            os.utime(os.path.join(self.dir, uid))
        except OSError:
            pass

    def finish(self, uid):
        with open(os.path.join(self.dir, uid + ".done"), "w") as f:
            f.write(str(time.time()))


class HourMeter:
    """Hours of finished episodes per source in the nominal train + val splits (refreshed at most every 60 s)."""

    def __init__(self, *, root, control_hz):
        self.root, self.control_hz, self._t, self._hours = root, control_hz, 0, {}

    def hours(self):
        if time.time() - self._t > 60:
            hours = {}
            for split in ("train", "val"):
                path = os.path.join(self.root, split, "episodes.jsonl")
                if os.path.exists(path):
                    with open(path) as f:
                        for line in f:
                            r = json.loads(line)
                            hours[r["source"]] = hours.get(r["source"], 0) + r["steps"] / self.control_hz / 3600
            self._hours, self._t = hours, time.time()
        return self._hours


def write_episode(*, root, unit, robot, episode_id, source, arrays, frames, summary, extra, shift, level):
    split = unit["split"]
    rows = dict(arrays)
    rows["instruction"] = summary["instruction"]
    record = {"robot": robot, "task": unit["task"], "split": split, "source": source, "unit_id": unit["unit_id"],
              "start": unit.get("start"), "group": unit["unit_id"] if unit["kind"] == "group" else None,
              "noise_level": None, "replay_type": None, "shift": shift, "shift_level": level, **summary, **extra}
    RolloutDir(path=os.path.join(root, split)).write(episode_id=episode_id, arrays=rows, frames=frames,
                                                       fps=summary["fps"], summary=record)


def run_unit(*, unit, backend, robot, rcfg, cfg, root, shift_factory, shift, level, written, claims):
    rng = np.random.default_rng(zlib.crc32(unit["unit_id"].encode()))
    sigmas = rcfg["noise_sigmas"]
    # one shift object per unit: shifts that add bodies (distractors) build them in before_reset, so the saved start
    # state (groups, replays) already contains them
    sh = shift_factory() if shift_factory else None
    if sh is not None and hasattr(sh, "before_reset"):
        sh.before_reset(backend, task=unit["task"])
    common = dict(root=root, unit=unit, robot=robot, shift=shift, level=level)

    def episode(episode_id, source, extra, **kw):
        if episode_id in written:
            return None
        arrays, frames, summary = run_episode(backend=backend, unit=unit_k, rng=rng, shift=sh, **kw)
        write_episode(episode_id=episode_id, source=source, arrays=arrays, frames=frames, summary=summary,
                      extra=extra, **common)
        claims.touch(unit["unit_id"])
        return arrays

    kind = unit["kind"]
    if kind == "demo":
        import h5py  # demo replays only (LIBERO); not in every simulator environment

        with h5py.File(unit["demo_file"]) as f:
            d = f["data"][unit["demo_key"]]
            actions, state0 = d["actions"][()], d["states"][()][0]
        unit_k = dict(unit)
        obs = backend.reset_to_state(task=unit["task"], state=state0)
        episode(unit["unit_id"], "demo", {}, actions=actions, start_obs=obs)
        if unit.get("replay"):
            obs = backend.reset_to_state(task=unit["task"], state=state0)
            pert, meta = perturb(actions=actions, kind=unit["replay"], rng=rng, robot=robot,
                                 scale=backend.noise_scale(), sigmas=sigmas, control_hz=backend.control_hz)
            episode(unit["unit_id"] + "_r", "replay", {"replay_type": unit["replay"], "replay_meta": meta},
                    actions=pert, start_obs=obs)
    elif kind == "idle":
        unit_k = dict(unit)
        obs = backend.reset(task=unit["task"], start=unit["start"])
        actions = [backend.idle_action()] * int(rng.integers(3, 7) * backend.control_hz)
        episode(unit["unit_id"], "idle", {}, actions=actions, start_obs=obs)
    elif kind == "policy":
        unit_k = dict(unit, noise_sigma=sigmas[unit["noise_level"]])
        obs = backend.reset(task=unit["task"], start=unit["start"])
        state0 = backend.save_state()
        arrays = episode(unit["unit_id"], "policy", {"noise_level": unit["noise_level"],
                                                     "noise_sigma": sigmas[unit["noise_level"]]}, start_obs=obs)
        if unit.get("replay") and (arrays is not None or unit["unit_id"] + "_r" not in written):
            if arrays is None:   # parent written by an evicted worker: its commands are on disk
                arrays = dict(np.load(os.path.join(root, unit["split"], unit["unit_id"] + ".npz")))
            obs = backend.restore_state(state0)
            pert, meta = perturb(actions=arrays["action_cmd"], kind=unit["replay"], rng=rng, robot=robot,
                                 scale=backend.noise_scale(), sigmas=sigmas, control_hz=backend.control_hz)
            unit_k = dict(unit)
            episode(unit["unit_id"] + "_r", "replay", {"replay_type": unit["replay"], "replay_meta": meta},
                    actions=pert, start_obs=obs)
    elif kind == "group":
        obs = backend.reset(task=unit["task"], start=unit["start"])
        state0 = backend.save_state()
        for lv in unit["levels"]:
            unit_k = dict(unit, noise_sigma=sigmas[lv])
            obs = backend.restore_state(state0)
            episode(f"{unit['unit_id']}_n{lv}", "policy", {"noise_level": lv, "noise_sigma": sigmas[lv]}, start_obs=obs)
    else:
        raise ValueError(kind)


@click.command()
@click.option("--robot", required=True)
@click.option("--kind", type=click.Choice(["nominal", "test", "calib"]), default="nominal")
@click.option("--shift", default=None)
@click.option("--level", default=None, type=int)
@click.option("--groups_per_task", default=None, type=int, help="test sets: groups per task (default: all test starts)")
@click.option("--calib_units", default=200, type=int, help="calibration pools: policy episodes")
@click.option("--port", default=5555, type=int)
@click.option("--stale_min", default=20, type=int)
@click.option("--plan_only", is_flag=True)
def main(robot, kind, shift, level, groups_per_task, calib_units, port, stale_min, plan_only):
    parameters = load_parameters()
    spec = dict(get_robot(robot=robot, parameters=parameters))
    cfg = parameters["data"]
    rcfg = cfg["robots"][robot]
    spec["data"] = rcfg
    backend = make_backend(robot=robot, spec=spec, parameters=parameters, port=port)
    name = dataset_name(kind=kind, shift=shift, level=level)
    root = dataset_dir(parameters=parameters, robot=robot, dataset=name)
    if kind == "nominal":
        units, budgets = plan_nominal(backend=backend, robot=robot, parameters=parameters)
    elif kind == "test":
        units, budgets = plan_groups(backend=backend, robot=robot, parameters=parameters, split="test",
                                     starts_key="test", groups_per_task=groups_per_task), None
    else:
        n_tasks = len(backend.tasks(split="calib"))
        units = plan_groups(backend=backend, robot=robot, parameters=parameters, split="calib", starts_key="calib",
                            groups_per_task=-(-calib_units // n_tasks))   # just enough starts per task
        units = [dict(u, kind="policy", noise_level=0, replay=None) for u in units]
        units.sort(key=lambda u: (u["start"], u["task"]))   # round-robin over tasks, so a short pool covers them all
        units = units[:calib_units] if len(units) > calib_units else units
        budgets = None
    write_plan(root=root, units=units, budgets=budgets, config={"robot": robot, "kind": kind, "shift": shift,
               "level": level, "data": rcfg}, parameters=parameters)
    if plan_only:
        return
    shift_factory = None
    if shift is not None:
        from pwm.shifts import make_shift
        shift_factory = lambda: make_shift(robot=robot, name=shift, level=level)
    claims = Claims(root=root, stale_min=stale_min)
    meter = HourMeter(root=root, control_hz=backend.control_hz)
    source_of = {"policy": "policy", "demo": "demo", "idle": "idle"}
    n_done = 0
    for unit in units:
        uid = unit["unit_id"]
        if claims.done(uid):
            continue
        if budgets is not None:
            hours = meter.hours()
            if all(hours.get(s, 0) >= b for s, b in budgets.items()):
                log_info(f"{robot} {name}: all hour budgets met {hours}", parameters=parameters)
                break
            src = source_of[unit["kind"]]
            if hours.get(src, 0) >= budgets[src]:
                continue
            if unit.get("replay") and hours.get("replay", 0) >= budgets["replay"]:
                unit = dict(unit, replay=None)
        if not claims.claim(uid):
            continue
        written = set()
        index = os.path.join(root, unit["split"], "episodes.jsonl")
        if os.path.exists(index):
            with open(index) as f:
                written = {json.loads(l)["episode_id"] for l in f if uid in l}
        try:
            run_unit(unit=unit, backend=backend, robot=robot, rcfg=rcfg, cfg=cfg, root=root,
                     shift_factory=shift_factory, shift=shift, level=level, written=written, claims=claims)
        except Exception as e:  # a broken unit must not stop the worker; it stays claimed and is retried when stale
            log_warn(f"{robot} {name} unit {uid} failed: {type(e).__name__}: {e}", parameters=parameters)
            continue
        claims.finish(uid)
        n_done += 1
        if n_done % 20 == 0:
            log_info(f"{robot} {name}: {n_done} units done by this worker; hours {meter.hours()}", parameters=parameters)
    backend.close()
    log_info(f"{robot} {name}: worker finished ({n_done} units)", parameters=parameters)


if __name__ == "__main__":
    main()
