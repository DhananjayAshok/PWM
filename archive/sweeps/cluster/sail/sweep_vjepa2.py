"""Dev-box orchestrator of the V-JEPA 2-AC sweep on Sail. Keeps the sweep going without anyone watching; light
(only `sky jobs queue/launch/cancel` calls and small file reads). Every 10 min:

  1. reaper: cancels sweep jobs that are still queued/running although all their runs have finished
     (STOPPED.json + rollout_eval.json in every run directory);
  2. phase 1 (configs/sweeps/vjepa2.yaml, grid runs): for every robot whose nominal data is complete, launches each
     unfinished run that has no live job -- `base` alone on a GPU, the others packed 3 per GPU -- and relaunches runs
     whose job vanished or failed without STOPPED.json, at most 3 tries per run;
  3. phase 2: once all of a robot's grid runs are finished (or out of tries), launches the extra seeds (SEEDS,
     <run>-seed<N>) of its two best grid runs by val/loss_ar, packed 3 per GPU;
  4. refreshes results_dir/sweeps/vjepa2/<robot>.md.

State (run -> job id, tries) and the log are in results_dir/logs/sweep_vjepa2/. Start it detached from the project root:
    setsid nohup setup/.venv/bin/python cluster/sail/sweep_vjepa2.py >> <results_dir>/logs/sweep_vjepa2/orchestrator.log \\
        2>&1 < /dev/null & disown
Stop: kill the PID in results_dir/logs/sweep_vjepa2/orchestrator.pid.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pwm.utils import load_parameters  # noqa: E402

ROBOTS = ["franka_libero", "gr1_robocasa", "widowx_bridge", "google_fractal"]
CONTROL_HZ = {"franka_libero": 20, "gr1_robocasa": 20, "widowx_bridge": 5, "google_fractal": 3}
DATASET = "nominal"
PRIORITY = "p2"
PACK = 3
MAX_TRIES = 3
SEEDS = [1, 2]  # extra seeds of the two best grid runs
PERIOD_S = 600
SELECT = "val/loss_ar"

P = load_parameters()
LOG_DIR = os.path.join(P["results_dir"], "logs", "sweep_vjepa2")
STATE = os.path.join(LOG_DIR, "state.json")
RUNS = yaml.safe_load(open(os.path.join(P["project_root"], "configs", "sweeps", "vjepa2.yaml")))["runs"]
GRID = list(RUNS)


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} {msg}", flush=True)


def sh(cmd, timeout=900):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=P["project_root"])
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return ""


def live_jobs() -> dict:
    """{job_id: name} of jobs not yet finished (None when the queue cannot be read)."""
    out = sh("sky jobs queue --skip-finished -a", timeout=300)
    if "Managed jobs" not in out and "No in-progress" not in out:
        return None
    return {int(m[1]): m[2] for m in re.finditer(r"^(\d+)\s+\S+\s+(\S+)", out, re.M)}


def run_dir(robot, key):
    return os.path.join(P["storage_dir"], "checkpoints", "vjepa2", robot, f"{DATASET}-{key}")


def finished(robot, key):
    d = run_dir(robot, key)
    return os.path.exists(os.path.join(d, "STOPPED.json")) and os.path.exists(os.path.join(d, "rollout_eval.json"))


def best_val(robot, key):
    path = os.path.join(run_dir(robot, key), "STOPPED.json")
    if not os.path.exists(path):
        return None
    hist = json.load(open(path))["history"]
    return min(h[SELECT] for h in hist) if hist else None


def extra_seeds(runs):
    return [f"{r}-seed{s}" for r in runs for s in SEEDS]


def data_ready(robot):
    root = os.path.join(P["data_dir"], robot, DATASET)
    try:
        budget = sum(yaml.safe_load(open(os.path.join(root, "plan.yaml")))["budgets_hours"].values())
    except FileNotFoundError:
        return False
    hours = 0.0
    for split in ("train", "val"):
        path = os.path.join(root, split, "episodes.jsonl")
        if os.path.exists(path):
            rows = {}
            for line in open(path):
                if line.strip():
                    r = json.loads(line)
                    rows[r["episode_id"]] = r
            hours += sum(r["steps"] for r in rows.values()) / CONTROL_HZ[robot] / 3600
    return hours >= 0.93 * budget


def launch(robot, runs):
    """One job for `runs`; returns (job id, name) or None."""
    if len(runs) == 1:
        name = f"vj2-{robot}-{runs[0]}".replace("_", "-")
        cmd = (f"bash cluster/sail/launch.sh cluster/sail/train_vjepa2.yaml {name} --priority {PRIORITY} "
               f"--env ROBOT={robot} --env DATASET={DATASET} --env RUN={runs[0]} "
               f"--env EXPERIMENT_TAG={name} --env WANDB_RUN_ID={name}")
    else:
        name = f"vj2-{robot.replace('_', '-')}-pack-{hashlib.md5(' '.join(runs).encode()).hexdigest()[:6]}"
        cmd = (f"bash cluster/sail/launch.sh cluster/sail/train_vjepa2_pack.yaml {name} --priority {PRIORITY} "
               f"--env ROBOT={robot} --env DATASET={DATASET} --env RUNS='{' '.join(runs)}' "
               f"--env EXPERIMENT_TAG={name}")
    out = sh(f"HF_HOME={os.environ.get('HF_HOME', '/persist/cache/huggingface')} {cmd}")
    m = re.search(r"Managed Job ID: (\d+)", out)
    log(f"launch {name} ({' '.join(runs)}): " + (f"job {m[1]}" if m else f"FAILED: {out[-300:]}"))
    return (int(m[1]), name) if m else None


def schedule(state, live, robot, runs, pack):
    """Launch the runs that are unfinished, have no live job and tries left."""
    todo = []
    for run in runs:
        sk = f"{robot}/{run}"
        st = state.setdefault(sk, {"job": None, "tries": 0})
        if finished(robot, run) or (st["job"] is not None and st["job"] in live):
            continue
        if st["tries"] >= MAX_TRIES:
            if not st.get("given_up"):
                log(f"giving up on {sk} after {st['tries']} tries")
                st["given_up"] = True
            continue
        todo.append(run)
    for i in range(0, len(todo), pack):
        chunk = todo[i : i + pack]
        res = launch(robot, chunk)
        for run in chunk:
            st = state[f"{robot}/{run}"]
            st["tries"] += 1
            if res:
                st["job"], st["name"] = res
        if res:
            live[res[0]] = res[1]


def reap(state, live):
    jobs = {}
    for sk, st in state.items():
        if st["job"] is not None:
            jobs.setdefault(st["job"], []).append(sk)
    for job, keys in jobs.items():
        if job in live and all(finished(*k.split("/", 1)) for k in keys):
            log(f"reaper: job {job} ({live[job]}) finished all runs but is still live: cancelling")
            sh(f"sky jobs cancel -y {job}", timeout=300)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "orchestrator.pid"), "w") as f:
        f.write(str(os.getpid()))
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    while True:
        live = live_jobs()
        if live is None:
            log("could not read the job queue; retrying next round")
        else:
            reap(state, live)
            for robot in ROBOTS:
                if not data_ready(robot):
                    continue
                schedule(state, live, robot, ["base"], 1)
                schedule(state, live, robot, [r for r in GRID if r != "base"], PACK)
                done = [r for r in GRID if finished(robot, r)]
                exhausted = all(finished(robot, r) or state.get(f"{robot}/{r}", {}).get("given_up") for r in GRID)
                if exhausted and done:
                    ranked = sorted(done, key=lambda r: best_val(robot, r) or float("inf"))
                    best = state.get(f"{robot}/best") or ranked[0]  # fixed once chosen
                    if state.get(f"{robot}/best") != best:
                        log(f"{robot}: best grid run {best} ({best_val(robot, best):.5f}); extra seeds of the top 2")
                        state[f"{robot}/best"] = best
                    second = next((r for r in ranked if r != best), None)
                    schedule(state, live, robot, extra_seeds([best] + ([second] if second else [])), PACK)
            with open(STATE + ".tmp", "w") as f:
                json.dump(state, f, indent=1)
            os.replace(STATE + ".tmp", STATE)
        for robot in ROBOTS:
            sh(f"setup/.venv/bin/python -m pwm.wm.vjepa2.sweep_table --robot {robot}", timeout=300)
        time.sleep(PERIOD_S)


if __name__ == "__main__":
    main()
