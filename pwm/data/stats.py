"""Dataset validator and stats: hours and success by source and noise level, schema checks on every episode.

    python -m pwm.data.stats --robot widowx_bridge [--dataset nominal] [--check_files]
"""
import json
import os
from collections import defaultdict

import click
import numpy as np

from pwm.data.units import dataset_dir
from pwm.utils import load_parameters

REQUIRED = ("action_cmd", "action_env", "state", "probe_state", "success")


def load_index(root):
    rows = []
    for split in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, split, "episodes.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def check_episode(*, root, row):
    """Problems with one episode's files (empty list = valid)."""
    base = os.path.join(root, row["split"], row["episode_id"])
    problems = []
    if not os.path.exists(base + ".npz") or not os.path.exists(base + ".mp4"):
        return ["missing npz/mp4"]
    d = np.load(base + ".npz")
    for k in REQUIRED:
        if k not in d:
            problems.append(f"missing {k}")
    if problems:
        return problems
    T = len(d["action_cmd"])
    if T != row["steps"] or len(d["state"]) != T + 1 or len(d["probe_state"]) != T + 1 or len(d["success"]) != T:
        problems.append("length mismatch")
    for k in ("action_cmd", "state", "probe_state"):
        if not np.all(np.isfinite(d[k])):
            problems.append(f"non-finite {k}")
    return problems


def summarise(*, rows, control_hz):
    by = defaultdict(lambda: [0, 0.0, 0])
    for r in rows:
        key = (r["split"], r["source"], r.get("noise_level"), r.get("replay_type"))
        b = by[key]
        b[0] += 1
        b[1] += r["steps"] / control_hz / 3600
        b[2] += int(r["success"])
    lines = [f"{'split':6} {'source':7} {'noise':5} {'replay':7} {'episodes':>8} {'hours':>7} {'success':>8}"]
    for key in sorted(by, key=str):
        n, h, s = by[key]
        lines.append(f"{key[0]:6} {key[1]:7} {str(key[2]):5} {str(key[3]):7} {n:8d} {h:7.2f} {s / max(n, 1):8.1%}")
    total = sum(v[1] for k, v in by.items() if k[0] in ("train", "val"))
    lines.append(f"nominal train+val hours: {total:.2f}")
    return "\n".join(lines)


@click.command()
@click.option("--robot", required=True)
@click.option("--dataset", default="nominal")
@click.option("--check_files", is_flag=True, help="also open every npz and check the schema")
def main(robot, dataset, check_files):
    from pwm.robots import get_robot
    parameters = load_parameters()
    hz = {"franka_libero": 20, "gr1_robocasa": 20, "widowx_bridge": 5, "google_fractal": 3}[robot]
    get_robot(robot=robot, parameters=parameters)
    root = dataset_dir(parameters=parameters, robot=robot, dataset=dataset)
    rows = load_index(root)
    print(f"{robot}/{dataset}: {len(rows)} episodes")
    print(summarise(rows=rows, control_hz=hz))
    if check_files:
        bad = {r["episode_id"]: p for r in rows if (p := check_episode(root=root, row=r))}
        print(f"schema problems: {len(bad)}")
        for k, v in list(bad.items())[:20]:
            print(" ", k, v)


if __name__ == "__main__":
    main()
