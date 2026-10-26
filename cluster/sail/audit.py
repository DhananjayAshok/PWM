"""Health audit of the unattended run, from the dev box (one `sky jobs queue` call, then file reads only):

  - queue: our jobs by family and state, duplicate live jobs (same name twice), jobs that failed recently;
  - shift suites: episodes written per robot (and in the last hour);
  - dev-box loops alive (reaper, watchdog, usage logger) and the reaper's recent cancels.
(The per-model sweep sections were removed with the sweeps on 2026-10-26.)

    setup/.venv/bin/python cluster/sail/audit.py
Prints a report; lines starting with "!!" need attention. A section that raises prints a "!!" line instead of cutting
the report short, and a complete report ends with "# audit complete".
"""
import collections
import functools
import glob
import os
import re
import subprocess
import sys
import time

import click

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pwm.utils import load_parameters

FAMILIES = (("pwm-gensh", "shift-suite"), ("ddre-", "dreamdojo-eval"), ("dd-", "dreamdojo"), ("vj2", "vjepa2"),
            ("dv3sw", "dreamerv3"), ("pwm-", "other"))
LIVE = ("PENDING", "STARTING", "RUNNING", "RECOVERING", "CANCELLING")
STATES = LIVE + ("SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_CONTROLLER", "CANCELLED")


def family(name: str):
    return next((f for prefix, f in FAMILIES if name.startswith(prefix)), None)


def age_min(path: str) -> float:
    return (time.time() - os.path.getmtime(path)) / 60


@functools.cache
def queue() -> list[dict]:
    proc = subprocess.run("sky jobs queue -a", check=False, shell=True, capture_output=True, text=True, timeout=300)
    rows = []
    for line in proc.stdout.splitlines():
        f = line.split()
        if len(f) > 4 and f[0].isdigit():  # ID TASK WORKSPACE NAME ...
            state = next((x for x in f[4:] if x in STATES), "?")
            rows.append({"id": int(f[0]), "name": f[3], "state": state, "line": line})
    if not rows:
        raise RuntimeError(f"no jobs parsed from `sky jobs queue -a` (rc {proc.returncode}): {proc.stderr[-200:]}")
    return rows


def audit_queue(*, P, since_id):
    rows = [r for r in queue() if family(r["name"])]
    if not since_id and rows:
        since_id = sorted(r["id"] for r in rows)[-400] if len(rows) > 400 else 0
    rows = [r for r in rows if r["id"] >= since_id]
    counts = collections.Counter((family(r["name"]), r["state"]) for r in rows)
    print("## queue (our jobs, id >=", since_id, ")")
    for fam in sorted({f for f, _ in counts}):
        print(f"  {fam:15s} " + ", ".join(f"{n} {s}" for (f, s), n in sorted(counts.items()) if f == fam))
    live = collections.defaultdict(list)
    for r in rows:
        if r["state"] in LIVE and r["state"] != "CANCELLING":
            live[r["name"]].append(r["id"])
    for name, ids in sorted(live.items()):
        if len(ids) > 1:
            print(f"!! duplicate live jobs {name}: {sorted(ids)}")
    for r in rows:
        if r["state"].startswith("FAILED"):
            print(f"!! failed job {r['id']} {r['name']} ({r['state']})")





def audit_shifts(*, P, since_id):
    print("\n## shift suites (episodes written)")
    for robot in ("franka_libero", "gr1_robocasa", "widowx_bridge", "google_fractal"):
        files = glob.glob(os.path.join(P["storage_dir"], "data", robot, "shift-*", "*", "episodes.jsonl"))
        total = 0
        for f in files:
            with open(f) as fh:
                total += sum(1 for _ in fh)
        recent = sum(1 for f in files if age_min(f) < 60)
        print(f"  {robot:15s} {len({f.split('/')[-3] for f in files}):3d} shift-level sets, {total:6d} episodes, "
              f"{recent} sets written in the last hour")


def audit_loops(*, P, since_id):
    print("\n## dev-box loops")
    procs = subprocess.run("ps -eo cmd", check=False, shell=True, capture_output=True, text=True).stdout
    for name, pat in (("reaper", r"reaper\.sh"), ("watchdog", r"watchdog\.sh"), ("usage logger", r"gpu-usage-logger")):
        alive = re.search(pat, procs) is not None
        print(f"{'  ' if alive else '!!'}{name:13s} {'alive' if alive else 'NOT RUNNING'}")
    reaper = os.path.join(P["results_dir"], "logs", "reaper.log")
    if os.path.exists(reaper):
        with open(reaper) as f:
            cancels = [l for l in f if " cancel " in l][-5:]
        print("  reaper, last cancels: " + ("; ".join(l.split(" cancel ")[0][-5:] + " " + l.split(" cancel ")[1].split(":")[0]
                                                    for l in cancels) or "none"))


@click.command()
@click.option("--since_id", default=0, help="Only jobs with at least this id (0: the last 400 of ours)")
def main(since_id):
    P = load_parameters()
    print(f"# audit {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n")
    for section in (audit_queue, audit_shifts, audit_loops):
        try:
            section(P=P, since_id=since_id)
        except Exception as e:  # noqa: BLE001 - a broken section must show up, not truncate the report
            print(f"!! audit section {section.__name__} failed: {type(e).__name__}: {e}")
    print("\n# audit complete")


if __name__ == "__main__":
    main()
