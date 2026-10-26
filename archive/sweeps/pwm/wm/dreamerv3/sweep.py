"""Sweep bookkeeping shared by the trainer, the results table and the orchestrator (no torch imports: it also runs in
the core environment).

Run summaries: <storage_dir>/sweeps/dreamerv3/<robot>/<run_name>.json (written by train.py). Once all of a robot's
grid runs have stopped, ``select`` picks the one with the lowest fixed selection metric (val/openloop_mse64 at its best
checkpoint) and writes <storage_dir>/sweeps/dreamerv3/<robot>/best_grid.json; the ``best_grid`` sweep run builds on that
run for this robot, ``second_grid`` on the runner-up.

Seeds: ``<run>-seed<N>`` is any sweep run trained with seed N (the plain name is seed 0); the ``seeds`` group trains
the extra seeds of ``best_grid`` and ``second_grid``.

    python -m pwm.wm.dreamerv3.sweep select --robot franka_libero     # prints the chosen run, or nothing if not ready
    python -m pwm.wm.dreamerv3.sweep done --robot franka_libero --run baseline   # prints 1 if stopped
"""
import datetime
import json
import os
import re
from typing import Optional

import click
import yaml

from pwm.utils import load_parameters, log_info


def sweep_dir(*, parameters: dict, robot: str) -> str:
    return os.path.join(parameters["storage_dir"], "sweeps", "dreamerv3", robot)


def load_runs(*, parameters: dict) -> dict:
    with open(os.path.join(parameters["project_root"], "configs", "sweeps", "dreamerv3.yaml")) as f:
        return yaml.safe_load(f)


def summary(*, parameters: dict, robot: str, run: str, dataset: str = "nominal") -> Optional[dict]:
    name = f"dv3-{robot}-{run}" if dataset == "nominal" else f"dv3-{robot}-{dataset}-{run}"
    try:
        with open(os.path.join(sweep_dir(parameters=parameters, robot=robot), name + ".json")) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def split_seed(name: str) -> tuple[str, int]:
    """``best_grid-seed2`` -> (``best_grid``, 2); a plain name is seed 0."""
    m = re.fullmatch(r"(.+)-seed(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def _grid_choice(*, parameters: dict, robot: str, rank: int) -> Optional[str]:
    try:
        with open(os.path.join(sweep_dir(parameters=parameters, robot=robot), "best_grid.json")) as f:
            return json.load(f)["ranking"][rank][0]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError):
        return None


def best_grid_run(*, parameters: dict, robot: str) -> Optional[str]:
    """The chosen grid run for this robot (status file), or None before the grid has finished."""
    return _grid_choice(parameters=parameters, robot=robot, rank=0)


def base_of(*, runs: dict, name: str, robot: str, parameters: dict) -> Optional[str]:
    """A run's base: for ``best_grid`` / ``second_grid`` the status file's choice (else the yaml base); a mapping base
    is per robot."""
    for alias, rank in (("best_grid", 0), ("second_grid", 1)):
        if name == alias:
            chosen = _grid_choice(parameters=parameters, robot=robot, rank=rank)
            if chosen:
                return chosen
    base = (runs.get(name) or {}).get("base")
    return base.get(robot) if isinstance(base, dict) else base


def seed_scores(*, parameters: dict, robot: str, run: str, seeds: list[int]) -> Optional[list[float]]:
    """Selection metric of a run at each seed (``best_grid`` / ``second_grid`` at seed 0 = that grid run itself); None
    if any is missing."""
    out = []
    for seed in seeds:
        if seed == 0:
            rank = {"best_grid": 0, "second_grid": 1}.get(run)
            name = run if rank is None else _grid_choice(parameters=parameters, robot=robot, rank=rank)
        else:
            name = f"{run}-seed{seed}"
        s = summary(parameters=parameters, robot=robot, run=name)
        if not s or not s.get("stop_reason") or s.get("best") is None:
            return None
        out.append(float(s["best"]))
    return out


def select(*, parameters: dict, robot: str) -> Optional[str]:
    sweep = load_runs(parameters=parameters)
    grid = sweep["groups"]["grid"]
    rows = {r: summary(parameters=parameters, robot=robot, run=r) for r in grid}
    if any(s is None or not s.get("stop_reason") for s in rows.values()):
        return None
    scored = {r: s["best"] for r, s in rows.items() if s.get("best") is not None and s["best"] == s["best"]}
    chosen = min(scored, key=scored.get)
    path = os.path.join(sweep_dir(parameters=parameters, robot=robot), "best_grid.json")
    record = {"run": chosen, "selection": "val/openloop_mse64", "value": scored[chosen], "best_step": rows[chosen]["best_step"],
              "ranking": sorted(([r, v] for r, v in scored.items()), key=lambda x: x[1]),
              "chosen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(path + ".tmp", "w") as f:
        json.dump(record, f, indent=1)
    os.replace(path + ".tmp", path)
    log_info(f"dreamerv3 {robot}: best grid run {chosen} (openloop_mse64 {scored[chosen]:.5f}); ranking "
             + ", ".join(f"{r} {v:.5f}" for r, v in record["ranking"]), parameters=parameters)
    return chosen


@click.group()
def cli():
    pass


@cli.command("select")
@click.option("--robot", required=True)
def select_cmd(robot):
    parameters = load_parameters()
    chosen = best_grid_run(parameters=parameters, robot=robot) or select(parameters=parameters, robot=robot)
    if chosen:
        print(chosen)


@cli.command("done")
@click.option("--robot", required=True)
@click.option("--run", required=True)
def done_cmd(robot, run):
    s = summary(parameters=load_parameters(), robot=robot, run=run)
    print(int(bool(s and s.get("stop_reason"))))


if __name__ == "__main__":
    cli()
