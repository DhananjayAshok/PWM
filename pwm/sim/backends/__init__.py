"""Simulator backends: one per robot, a common interface for data generation and evaluation (see base.Backend).
Each module imports its simulator lazily, so this package imports in any environment."""

BACKENDS = {
    "franka_libero": "pwm.sim.backends.libero:LiberoBackend",
    "gr1_robocasa": "pwm.sim.backends.robocasa_gr1:RoboCasaGR1Backend",
    "widowx_bridge": "pwm.sim.backends.simpler:SimplerBackend",
    "google_fractal": "pwm.sim.backends.simpler:SimplerBackend",
}


def make_backend(*, robot, spec, parameters, port=5555):
    import importlib
    module, cls = BACKENDS[robot].split(":")
    return getattr(importlib.import_module(module), cls)(robot=robot, spec=spec, parameters=parameters, port=port)
