"""Shift registry: names, per-robot support, level ladders (configs/shifts.yaml) and their translation from the
screening robot (franka_libero) to the other robots in physically equivalent units."""
from typing import Any, Optional

from pwm.shifts import action as act
from pwm.shifts.base import NullShift, Shift

BACKENDS = {"franka_libero": "libero", "gr1_robocasa": "robocasa", "widowx_bridge": "simpler", "google_fractal": "simpler"}
ACTION_SHIFTS = {"latency": act.Latency, "latency_jitter": act.LatencyJitter, "frame_rotation": act.FrameRotation}


def _config(parameters):
    from pwm.utils.parameter_handling import load_parameters
    return load_parameters(parameters)["shifts"]


def _backend_shifts(robot):
    backend = BACKENDS[robot]
    if backend == "libero":
        from pwm.shifts import backend_libero as b
    elif backend == "robocasa":
        from pwm.shifts import backend_robocasa as b
    else:
        from pwm.shifts import backend_simpler as b
    return b


def shift_names(*, parameters=None) -> list[str]:
    return list(_config(parameters)["suite"])


def unsupported_reason(*, robot: str, name: str, parameters=None) -> Optional[str]:
    spec = _config(parameters)["suite"][name]
    return (spec.get("unsupported") or {}).get(robot)


def support_matrix(*, parameters=None) -> dict:
    cfg = _config(parameters)
    return {name: {robot: (unsupported_reason(robot=robot, name=name, parameters=parameters) or "ok")
                   for robot in cfg["robot_facts"]} for name in cfg["suite"]}


def translate(*, robot: str, name: str, value: Any, facts: dict, ladder_facts: dict) -> Any:
    """A screening-robot level value expressed for `robot`."""
    if isinstance(value, list):
        value = tuple(value)
    rule = name
    if rule in ("latency", "latency_jitter"):  # milliseconds -> whole control steps (at least one)
        return round(value)  # resolved to steps in make_shift (needs the whole ladder to keep levels distinct)
    if rule == "payload":  # same fraction of the robot's rated payload
        return value / ladder_facts["payload_kg"] * facts["payload_kg"]
    return value


def _ms_ladder_to_steps(ms_ladder, hz):
    """Ladder of milliseconds -> strictly increasing control steps (>= 1); slow-control robots get coarser steps."""
    steps, last = [], 0
    for ms in ms_ladder:
        s = max(1, int(round(ms * hz / 1000.0)), last + 1)
        steps.append(s)
        last = s
    return steps


def make_shift(*, robot: str, name: str, level: int, parameters=None) -> Shift:
    """The shift `name` at `level` (1..5; 0 or name 'none' = no shift) for `robot`."""
    cfg = _config(parameters)
    facts = dict(cfg["robot_facts"][robot])
    if name in (None, "none") or level == 0:
        return NullShift(robot=robot, facts=facts)
    spec = cfg["suite"][name]
    reason = (spec.get("unsupported") or {}).get(robot)
    if reason:
        raise ValueError(f"shift {name} does not apply to {robot}: {reason}")
    ladder = spec["levels"]
    if not 1 <= level <= len(ladder):
        raise ValueError(f"{name} has levels 1..{len(ladder)}, got {level}")
    ladder_facts = cfg["robot_facts"][cfg["screening_robot"]]
    value = translate(robot=robot, name=name, value=ladder[level - 1], facts=facts, ladder_facts=ladder_facts)
    if name == "latency":
        steps = _ms_ladder_to_steps(ladder, facts["control_hz"])[level - 1]
        value = {"steps": steps, "ms": 1000.0 * steps / facts["control_hz"]}
    elif name == "latency_jitter":
        steps = _ms_ladder_to_steps(ladder, facts["control_hz"])[level - 1]
        value = {"max_steps": steps, "ms": 1000.0 * steps / facts["control_hz"]}
    if name in ACTION_SHIFTS:
        cls = ACTION_SHIFTS[name]
    else:
        cls = _backend_shifts(robot).SHIFTS[name]
    return cls(robot=robot, name=name, level=level, value=value, facts=facts)


def make_shift_value(*, robot: str, name: str, value: Any, parameters=None) -> Shift:
    """A shift at an arbitrary value in the robot's own units (screening uses this before ladders exist)."""
    cfg = _config(parameters)
    facts = dict(cfg["robot_facts"][robot])
    if name == "latency":
        value = {"steps": int(max(1, round(value * facts["control_hz"] / 1000.0))), "ms": value}
    elif name == "latency_jitter":
        value = {"max_steps": int(max(1, round(value * facts["control_hz"] / 1000.0))), "ms": value}
    elif isinstance(value, list):
        value = tuple(value)
    cls = ACTION_SHIFTS.get(name) or _backend_shifts(robot).SHIFTS[name]
    return cls(robot=robot, name=name, level=-1, value=value, facts=facts)


def refresh_obs(robot: str, env):
    """Re-render the first observation after a shift's on_reset (robot-specific observation path)."""
    return _backend_shifts(robot).refresh_obs(env)
