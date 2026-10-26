"""Results table of the DreamerV3 sweep for one robot: <results_dir>/sweeps/dreamerv3/<robot>.md, built from the run
summaries the trainer keeps in <storage_dir>/sweeps/dreamerv3/<robot>/<run_name>.json (runs no longer in
configs/sweeps/dreamerv3.yaml, e.g. the dropped trick runs, are left out). Runs in the core environment.

    python -m pwm.wm.dreamerv3.sweep_table --robot franka_libero
"""
import glob
import json
import os

import click
import yaml

from pwm.utils import load_parameters, log_info
from pwm.wm.dreamerv3.sweep import base_of, best_grid_run, load_runs, seed_scores, split_seed


def _fmt(v, spec=".4f"):
    return "-" if v is None else format(v, spec)


def changes(*, runs: dict, name: str, robot: str, parameters: dict) -> str:
    """The run's overrides relative to the baseline, flattened (chain of bases included)."""
    seen, seed = split_seed(name)
    out = {"seed": seed} if seed else {}
    while seen:
        run = runs.get(seen) or {}
        for k, v in (run.get("overrides") or {}).items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    out.setdefault(f"{k}.{k2}" if k != "overrides" else k2, v2)
            else:
                out.setdefault(k, v)
        seen = base_of(runs=runs, name=seen, robot=robot, parameters=parameters)
    return ", ".join(f"{k}={v}" for k, v in sorted(out.items())) or "(baseline)"


def seed_section(*, parameters: dict, robot: str) -> list[str]:
    """Selection metric over seeds (0 + seeds_per_run) of the grid's two best runs."""
    groups = load_runs(parameters=parameters)["groups"]
    seeds = [0] + list(groups.get("seeds_per_run", []))
    names = groups["seeds"]
    rows = []
    for name in names:
        v = seed_scores(parameters=parameters, robot=robot, run=name, seeds=seeds)
        if name == "second_grid" and v is None:
            continue
        if v is None:
            rows.append(f"| {name} | - | - | - | incomplete |")
            continue
        mean = sum(v) / len(v)
        sd = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        rows.append(f"| {name} | {mean:.5f} | {sd:.5f} | {max(v):.5f} | {', '.join(f'{x:.5f}' for x in v)} |")
    return ["", f"## Over seeds {seeds}", "",
            "Grid runs have seed 0 only; `best_grid` / `second_grid` = the grid's two best runs with their extra "
            "seeds, to compare grid differences with seed noise.", "",
            "| run | mean MSE@64 | sd | worst seed | per seed |", "|---|---|---|---|---|", *rows]


@click.command()
@click.option("--robot", required=True)
@click.option("--dataset", default="nominal", help="Only runs on this dataset; others than nominal get <robot>-<dataset>.md")
def main(robot, dataset):
    parameters = load_parameters()
    with open(os.path.join(parameters["project_root"], "configs", "sweeps", "dreamerv3.yaml")) as f:
        runs = yaml.safe_load(f)["runs"]
    rows = []
    for path in sorted(glob.glob(os.path.join(parameters["storage_dir"], "sweeps", "dreamerv3", robot, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if not r.get("sweep_run") or r.get("dataset") != dataset or split_seed(r["sweep_run"])[0] not in runs:
            continue
        final = r.get("final") or {}
        best = next((h for h in r.get("history", []) if h["step"] == r.get("best_step")), {})
        rows.append({
            "run": r["sweep_run"], "changes": changes(runs=runs, name=r["sweep_run"], robot=robot, parameters=parameters),
            "status": "done" if r.get("stop_reason") else f"running ({r.get('step')})",
            "best": r.get("best"), "best_step": r.get("best_step"), "params": r.get("n_params", 0) / 1e6,
            "base": best.get("val/base_loss"), "image": best.get("val/image_loss"),
            "success": best.get("val/reward_loss"), "kl": best.get("val/kl_term"),
            "mse": final.get("openloop_mse_native"), "lpips": final.get("openloop_lpips_native"),
            "sens": final.get("action_sensitivity_mse"), "gpu_h": r.get("gpu_h", 0.0)})
    rows.sort(key=lambda x: (x["best"] is None, x["best"] if x["best"] is not None else 0))
    lines = [f"# DreamerV3 sweep: {robot}" + ("" if dataset == "nominal" else f" ({dataset} data)"), "",
             "Selection (fixed in advance): **open-loop MSE @64 px** of the 4.8 s prediction from 1 real frame on the fixed val episodes (all of them, up to 128; "
             "lower is better), at the run's best checkpoint. Base objective and its parts on 256 "
             "fixed val windows at that checkpoint. Native-resolution open-loop MSE / LPIPS and action sensitivity "
             "(MSE with other episodes' commands / MSE with the true commands; 1 = commands ignored) on 32 fixed val "
             "episodes. GPU-h: wall time x GPU share (runs are packed several per GPU). Configs: "
             "configs/sweeps/dreamerv3.yaml.", "",
             f"Best grid run (chosen automatically when the grid finished; its seed replicates build on it): "
             f"**{best_grid_run(parameters=parameters, robot=robot) or 'not chosen yet'}**", "",
             "| run | changes | status | **sel. MSE@64** | best step | params (M) | base obj. | image NLL | success NLL "
             "| KL | MSE native | LPIPS native | action sens. | GPU-h |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x['run']} | {x['changes']} | {x['status']} | {_fmt(x['best'], '.5f')} | {x['best_step']} | "
                     f"{x['params']:.1f} | {_fmt(x['base'], '.2f')} | {_fmt(x['image'], '.2f')} | "
                     f"{_fmt(x['success'], '.3f')} | {_fmt(x['kl'], '.3f')} | {_fmt(x['mse'], '.5f')} | "
                     f"{_fmt(x['lpips'], '.4f')} | {_fmt(x['sens'], '.2f')} | {x['gpu_h']:.2f} |")
    lines += seed_section(parameters=parameters, robot=robot) if dataset == "nominal" else []
    out = os.path.join(parameters["results_dir"], "sweeps", "dreamerv3")
    os.makedirs(out, exist_ok=True)
    name = robot if dataset == "nominal" else f"{robot}-{dataset}"
    with open(os.path.join(out, f"{name}.md.tmp"), "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(os.path.join(out, f"{name}.md.tmp"), os.path.join(out, f"{name}.md"))
    log_info(f"{len(rows)} runs -> {out}/{name}.md", parameters=parameters)


if __name__ == "__main__":
    main()
