"""Deployment shifts: one registry for every robot (configs/shifts.yaml, plans/shifts.md).

    from pwm.shifts import make_shift
    shift = make_shift(robot="widowx_bridge", name="latency", level=3)
    env = simpler_env.make(task, **shift.make_kwargs())      # scene-build shifts
    shift.before_reset(env); obs, info = env.reset(...); shift.on_reset(env)
    if shift.reobserve: obs = refresh_obs(robot, env)          # the first observation must show the shifted scene
    for t in ...: cmd = <policy command>; shift.on_step(env, t); obs, ... = env.step(shift.wrap_action(cmd, t))

Simulator modules (mujoco, sapien) are imported only by the backend that needs them, so this package can be imported in
every simulator environment.
"""
from pwm.shifts.base import Shift, NullShift
from pwm.shifts.registry import make_shift, shift_names, support_matrix, refresh_obs

__all__ = ["Shift", "NullShift", "make_shift", "shift_names", "support_matrix", "refresh_obs"]
