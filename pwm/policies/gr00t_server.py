"""Serve a GR00T policy for a robot (Isaac-GR00T's gr00t/eval/run_gr00t_server.py), in its gr00t_* environment.
Prints "Server is ready" once listening (callers wait for that line).

    python -m pwm.policies.gr00t_server --robot widowx_bridge [--port 5555]
"""
import sys

import click

from pwm.policies.gr00t_checkpoint import prepare
from pwm.robots import get_robot
from pwm.utils import load_parameters, log_info


@click.command()
@click.option("--robot", required=True)
@click.option("--port", default=5555, type=int)
def main(robot, port):
    parameters = load_parameters()
    spec = get_robot(robot=robot, parameters=parameters)
    model_path = prepare(repo=spec["policy"], revision=spec["policy_revision"], parameters=parameters)
    import torch
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA has no execution plans for some shapes on B300 (sm_103)
    log_info(f"torch {torch.__version__} cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)}",
             parameters=parameters)
    import tyro
    from gr00t.eval.run_gr00t_server import ServerConfig, main as serve
    sys.argv = [sys.argv[0], "--model-path", model_path, "--embodiment-tag", spec["embodiment_tag"],
                "--use-sim-policy-wrapper", "--port", str(port)]
    serve(tyro.cli(ServerConfig))


if __name__ == "__main__":
    main()
