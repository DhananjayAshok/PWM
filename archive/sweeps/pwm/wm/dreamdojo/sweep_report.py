"""Results table of the DreamDojo sweep for one robot: one row per run directory (progress.json written by the trainer
after every validation), sorted by best validation loss. Core environment (reads json only).

    python -m pwm.wm.dreamdojo.sweep_report --robot franka_libero
writes <results_dir>/sweeps/dreamdojo/<robot>.md and .json.
"""
import datetime
import glob
import json
import os

import click

from pwm.utils import load_parameters, log_info
from pwm.wm.dreamdojo.rollout_eval import VARIANT, summarise


def rows_for(*, root: str, robot: str) -> list[dict]:
    rows = []
    for info_path in glob.glob(os.path.join(root, "dd-*", "run_info.json")):
        run_dir = os.path.dirname(info_path)  # a run directory or its status mirror
        with open(info_path) as f:
            info = json.load(f)
        if info.get("robot") != robot or info.get("sweep_run", "none").startswith("smoke"):
            continue
        prog = {}
        if os.path.exists(os.path.join(run_dir, "progress.json")):
            with open(os.path.join(run_dir, "progress.json")) as f:
                prog = json.load(f)
        rows.append({"run": info.get("sweep_run"), "run_dir": run_dir, "overrides": info.get("overrides", {}),
                     "dataset": os.path.basename(info.get("dataset", "").rstrip("/")),
                     "best_val": prog.get("best_val"), "best_step": prog.get("best_step"),
                     "best_clips": prog.get("best_clips"), "step": prog.get("step"), "clips": prog.get("clips"),
                     "epochs": prog.get("epochs"),
                     "evals": len(prog.get("val_history", [])), "gpu_hours": prog.get("gpu_hours"),
                     "wall_hours": prog.get("wall_hours"), "world": prog.get("world"),
                     "status": ("done: " + prog["reason"]) if prog.get("done") else ("running" if prog else "started"),
                     "updated": prog.get("updated_utc"),
                     "rollout": summarise(variant_dir=os.path.join(run_dir, "rollout_eval", VARIANT))})
    return sorted(rows, key=lambda r: (r["best_val"] is None, r["best_val"] or 0))


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


@click.command()
@click.option("--robot", required=True)
@click.option("--root", "roots", multiple=True, help="Directories holding dd-* run directories or status mirrors "
                                                     "(repeatable); default <results_dir>/sweeps/dreamdojo/runs")
@click.option("--baseline", default="lr4e-5_b96_full")
def main(robot, roots, baseline):
    parameters = load_parameters()
    roots = roots or (os.path.join(parameters["results_dir"], "sweeps", "dreamdojo", "runs"),)
    rows = [r for root in roots for r in rows_for(root=root, robot=robot)]
    base = next((r["best_val"] for r in rows if r["run"] == baseline and r["best_val"] is not None), None)
    out = os.path.join(parameters["results_dir"], "sweeps", "dreamdojo")
    os.makedirs(out, exist_ok=True)
    lines = [f"# DreamDojo sweep: {robot}", "",
             f"Updated {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='minutes')}. Validation loss: the "
             "same fixed val clips, noise levels and seeds for every run (EMA weights); lower is better. A run stops after "
             "3 consecutive evaluations without improvement. Changes are relative to configs/world_models.yaml "
             "(lr 4e-5, effective batch 96, full fine-tune).", "",
             "Rollout columns: free-running first 4.8 s from frame 0 on 32 fixed val episodes (guidance-0 sampler): "
             "latent MSE and LPIPS vs the real future; action sensitivity = latent MSE with another episode's actions / "
             "with the true ones (higher = follows its actions more).", "",
             "| run | changes | best val loss | vs baseline | best at step (epoch) | rollout latent MSE | rollout LPIPS | "
             "action sensitivity | evals | GPU-h | wall h | GPUs | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        changes = ", ".join(f"{k}={v}" for k, v in r["overrides"].items()) or "-"
        rel = "-" if base is None or r["best_val"] is None else f"{(r['best_val'] - base) / base:+.1%}"
        # epoch at the best evaluation = best_clips / clips-per-epoch, and clips-per-epoch = clips / epochs
        ep = None if not (r["best_clips"] and r["clips"] and r["epochs"]) else r["best_clips"] * r["epochs"] / r["clips"]
        best_at = "-" if r["best_step"] is None else f"{r['best_step']} ({fmt(ep, '.2f')})"
        def roll(v):
            if not v:
                return "- | - | -"
            tag = "" if v["episodes"] >= 32 else f" ({v['episodes']} eps)"
            return f"{v['latent_mse']:.4f}{tag} | {v['lpips']:.3f} | {v['sensitivity']:.2f}"
        lines.append(f"| {r['run']} | {changes} | {fmt(r['best_val'], '.5f')} | {rel} | {best_at} | "
                     f"{roll(r['rollout'])} | {r['evals']} | {fmt(r['gpu_hours'], '.1f')} | "
                     f"{fmt(r['wall_hours'], '.1f')} | {r['world'] or '-'} | {r['status']} |")
    with open(os.path.join(out, f"{robot}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, f"{robot}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n".join(lines))
    log_info(f"sweep report for {robot}: {len(rows)} runs -> {out}/{robot}.md", parameters=parameters)


if __name__ == "__main__":
    main()
