"""Decides the next actions of the DreamDojo sweep (cluster/sail/sweep_dreamdojo.sh); reads status files only, no compute.

Stages (configs/sweeps/dreamdojo.yaml `groups`):
  1. grid     on the main robot: every grid run.
  2. seeds    once every grid run has finished: the two best grid points by the fixed validation loss, again at SEEDS
              (<run>-seed<N>: the same overrides, another training-data order), written into
              configs/sweeps/dreamdojo.yaml as named runs.
     confirm  the best grid point on the other robots.
  Every finished training run gets a rollout evaluation (pwm/wm/dreamdojo/rollout_eval.py).
A run is finished when its status mirror (<results_dir>/sweeps/dreamdojo/runs/<tag>/STOPPED) exists; a run whose job
vanished or failed without it is relaunched, at most `max_tries` times (tries counted in <state>).

    python -m pwm.wm.dreamdojo.sweep_plan --active <file with `sky jobs queue` output> --state <state.json>
prints one action per line:  TRAIN <robot> <run> <tag>  |  EVAL <robot> <run> <tag>  |  INFO <text>
"""
import json
import os

import click
import yaml

from pwm.utils import load_parameters
from pwm.wm.dreamdojo.rollout_eval import N_EPISODES, VARIANT, episode_files

SWEEP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                          "configs", "sweeps", "dreamdojo.yaml")
SEEDS = [1, 2]


def tag_of(robot: str, run: str) -> str:
    return f"dd-{robot}-nominal-{run}"


def eval_tag_of(robot: str, run: str) -> str:
    return f"ddre-{robot}-{run}"


def status_dir(parameters, tag: str) -> str:
    return os.path.join(parameters["results_dir"], "sweeps", "dreamdojo", "runs", tag)


def stopped(parameters, tag: str) -> dict | None:
    p = os.path.join(status_dir(parameters, tag), "STOPPED")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def eval_done(parameters, tag: str) -> bool:
    d = os.path.join(status_dir(parameters, tag), "rollout_eval", VARIANT)
    return len(episode_files(variant_dir=d)) >= N_EPISODES


def add_seed_runs(*, points: list[str], runs: dict) -> list[str]:
    """Append the seed replicates of `points` to the sweep file (names that exist are kept); returns them."""
    new, out = [], []
    for point in points:
        for s in SEEDS:
            run = f"{point}-seed{s}"
            out.append(run)
            if run not in runs:
                runs[run] = dict(runs[point] or {}, seed=s)
                new.append(f"  {run}: {json.dumps(runs[run])}  # added by sweep_plan: grid point seed replicate\n")
    if new:
        with open(SWEEP_FILE, "a") as f:
            f.writelines(new)
    return out


@click.command()
@click.option("--active", required=True, help="File with the current `sky jobs queue --skip-finished -a` output")
@click.option("--state", required=True, help="JSON file with launch counts (created if missing)")
@click.option("--max_tries", default=3, type=int)
def main(active, state, max_tries):
    parameters = load_parameters()
    with open(SWEEP_FILE) as f:
        sweep = yaml.safe_load(f)
    runs, groups = sweep["runs"], sweep["groups"]
    main_robot, others = groups["main_robot"], groups["confirm_robots"]
    queue = open(active).read()
    tries = json.load(open(state)) if os.path.exists(state) else {}
    is_active = lambda name: f" {name} " in f" {' '.join(queue.split())} "

    def want_train(robot, run):
        tag = tag_of(robot, run)
        if stopped(parameters, tag) or is_active(tag):
            return
        if tries.get(tag, 0) >= max_tries:
            print(f"INFO {tag}: gave up after {max_tries} tries")
            return
        tries[tag] = tries.get(tag, 0) + 1
        print(f"TRAIN {robot} {run} {tag}")

    def want_eval(robot, run):
        tag, etag = tag_of(robot, run), eval_tag_of(robot, run)
        if not stopped(parameters, tag) or eval_done(parameters, tag) or is_active(etag):
            return
        if tries.get(etag, 0) >= max_tries:
            print(f"INFO {etag}: gave up after {max_tries} tries")
            return
        tries[etag] = tries.get(etag, 0) + 1
        print(f"EVAL {robot} {run} {etag}")

    grid = groups["grid"]
    for run in grid:
        want_train(main_robot, run)
    finished = {run: stopped(parameters, tag_of(main_robot, run)) for run in grid}
    gave_up = {run for run in grid if tries.get(tag_of(main_robot, run), 0) >= max_tries and not finished[run]}
    if all(finished[r] or r in gave_up for r in grid):
        scored = {r: s["best_val"] for r, s in finished.items() if s and s.get("best_val") is not None}
        ranked = sorted(scored, key=scored.get)
        best, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        print(f"INFO grid done; best grid point {best} (val {scored[best]:.5f}), runner-up {second}")
        for run in add_seed_runs(points=[best] + ([second] if second else []), runs=runs):
            want_train(main_robot, run)
            want_eval(main_robot, run)
        for robot in others:
            want_train(robot, best)
            want_eval(robot, best)
    else:
        print(f"INFO grid: {sum(bool(v) for v in finished.values())}/{len(grid)} finished")
    for run in grid:
        want_eval(main_robot, run)
    with open(state + ".tmp", "w") as f:
        json.dump(tries, f, indent=1)
    os.replace(state + ".tmp", state)


if __name__ == "__main__":
    main()
