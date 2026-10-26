"""Plan the work units of a dataset (deterministic: the same config gives the same units, in the same order).

A unit is one start state and everything that must share it:
  policy  the strong policy at one noise level, plus (sometimes) a perturbed replay of its commands  [nominal]
  demo    a LIBERO human demo replayed in physics, plus a perturbed replay of it                      [nominal, Franka]
  idle    the robot holding still                                                                     [nominal]
  group   the strong policy at every counterfactual noise level from one identical start             [test sets]
Nominal units carry a split (train | val, by unit-id hash). Hour budgets are enforced by the workers, which take units
in this order and skip a source once its budget is met, so the order is a seeded shuffle that interleaves sources.
"""
import hashlib
import json
import os
import random

import yaml

from pwm.utils import log_info


def task_slug(task):
    return task.replace("gr1_unified/", "").replace("_GR1ArmsAndWaistFourierHands_Env", "").replace("/", "_t")


def unit_hash(unit_id, salt=""):
    return int(hashlib.sha256((salt + unit_id).encode()).hexdigest()[:12], 16) / 16 ** 12


def dataset_dir(*, parameters, robot, dataset):
    return os.path.join(parameters["data_dir"], robot, dataset)


def _libero_demos(*, parameters, tasks):
    """(task, hdf5 path, demo key, steps) for every human demo of the given LIBERO tasks."""
    import h5py
    from lerobot.envs.libero import _get_suite
    raw = os.path.join(parameters["data_dir"], "raw", "libero")
    out = []
    for task in tasks:
        suite, task_id = task.split("/")
        name = _get_suite(suite).tasks[int(task_id)].name
        path = os.path.join(raw, suite, f"{name}_demo.hdf5")
        with h5py.File(path) as f:
            for key in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1])):
                out.append((task, path, key, int(f["data"][key]["actions"].shape[0])))
    return out


def plan_nominal(*, backend, robot, parameters):
    cfg, rcfg = parameters["data"], parameters["data"]["robots"][robot]
    tasks = backend.tasks(split="train")
    lo, hi = rcfg["starts"]["train"]
    units, demo_hours = [], 0.0
    if rcfg.get("demos"):
        for task, path, key, steps in _libero_demos(parameters=parameters, tasks=tasks):
            uid = f"demo_{task_slug(task)}_{key}"
            units.append({"unit_id": uid, "kind": "demo", "task": task, "demo_file": path, "demo_key": key,
                          "replay": cfg["replay_types"][int(unit_hash(uid, "type") * 3)]})
            demo_hours += steps / backend.control_hz / 3600
    n_levels = len(rcfg["noise_sigmas"])
    p_replay = cfg["replay_hours"] / max(cfg["target_hours"] - cfg["replay_hours"] - cfg["idle_hours"], 0.1)
    for task in tasks:
        # few fixed starts (LIBERO: 45 per task): every start at every noise level; many seeds: one level per seed
        combos = [(lo + i, l) for i in range(hi - lo) for l in range(n_levels)] if hi - lo < 1000 else \
                 [(lo + i, None) for i in range(1000)]
        for start, level in combos:
            uid = f"pol_{task_slug(task)}_s{start}" + (f"_n{level}" if level is not None else "")
            if level is None:
                level = int(unit_hash(uid, "level") * n_levels)
            replay = None
            if rcfg["replay_parent"] == "policy" and unit_hash(uid, "replay") < p_replay:
                replay = cfg["replay_types"][int(unit_hash(uid, "type") * 3)]
            units.append({"unit_id": uid, "kind": "policy", "task": task, "start": start, "noise_level": level,
                          "replay": replay})
        for i in range(5):
            uid = f"idle_{task_slug(task)}_s{lo + i}"
            units.append({"unit_id": uid, "kind": "idle", "task": task, "start": lo + i})
    for u in units:
        u["split"] = "val" if unit_hash(u["unit_id"], "split") < cfg["val_fraction"] else "train"
    random.Random(0).shuffle(units)
    demo_hours = min(demo_hours, cfg["demo_hours"])
    budgets = {"idle": cfg["idle_hours"], "replay": cfg["replay_hours"], "demo": demo_hours,
               "policy": cfg["target_hours"] - cfg["idle_hours"] - cfg["replay_hours"] - demo_hours}
    if not rcfg.get("demos"):
        budgets.pop("demo")
    return units, budgets


def plan_groups(*, backend, robot, parameters, split, starts_key, groups_per_task=None):
    """Counterfactual groups: every task x start of the split's start range, all counterfactual noise levels."""
    rcfg = parameters["data"]["robots"][robot]
    lo, hi = rcfg["starts"][starts_key]
    if groups_per_task is not None:
        hi = min(hi, lo + groups_per_task)
    units = []
    for task in backend.tasks(split=split):
        for start in range(lo, hi):
            uid = f"grp_{task_slug(task)}_s{start}"
            units.append({"unit_id": uid, "kind": "group", "task": task, "start": start, "split": split,
                          "levels": list(parameters["data"]["counterfactual_levels"])})
    return units


def write_plan(*, root, units, budgets, config, parameters):
    """Write units.jsonl and plan.yaml (idempotent: an existing plan must match)."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "units.jsonl")
    text = "".join(json.dumps(u, sort_keys=True) + "\n" for u in units)
    if os.path.exists(path):
        with open(path) as f:
            if f.read() != text:
                raise RuntimeError(f"{path} exists with a different plan; use a new dataset name")
    else:
        with open(path + ".tmp", "w") as f:
            f.write(text)
        os.replace(path + ".tmp", path)
    with open(os.path.join(root, "plan.yaml"), "w") as f:
        yaml.safe_dump({"budgets_hours": budgets, "n_units": len(units), "config": config}, f, sort_keys=False)
    log_info(f"planned {len(units)} units in {root}; budgets {budgets}", parameters=parameters)
