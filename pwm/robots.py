"""Access to the robot registry in configs/robots.yaml (environments, policy, protocol, published success rates)."""
from typing import Any

from pwm.utils import log_error


def get_robot(*, robot: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Return the registry entry of one robot.

    :param robot: Robot name, a key of ``robots`` in configs/robots.yaml (e.g. ``widowx_bridge``).
    :param parameters: Loaded project parameters.
    :return: The robot's entry (``sim_env``, ``policy_env``, ``policy``, ``protocol``, ``tasks``, ``reference``, ...).
    """
    robots = parameters["robots"]
    if robot not in robots:
        log_error(f"Unknown robot {robot}; known: {sorted(robots)}", parameters=parameters)
    return robots[robot]
