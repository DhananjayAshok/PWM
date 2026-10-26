"""Client for Isaac-GR00T's policy server (gr00t/eval/run_gr00t_server.py), without importing gr00t: ZMQ REQ/REP with
msgpack + msgpack_numpy, the wire format of gr00t.policy.server_client.PolicyClient. Simulator environments use it to
query a policy that runs in its own environment (gr00t_n16 / gr00t_n17)."""
import time
from typing import Any, Optional

import msgpack
import msgpack_numpy
import zmq


class PolicyClient:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 120000):
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self._ctx = zmq.Context()
        self._sock = None
        self._connect(timeout_ms)

    def _connect(self, timeout_ms: int) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.connect(f"tcp://{self.host}:{self.port}")

    def call(self, *, endpoint: str, data: Optional[dict[str, Any]] = None) -> Any:
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self._sock.send(msgpack.packb(request, default=msgpack_numpy.encode))
        message = self._sock.recv()
        if message == b"ERROR":
            raise RuntimeError("policy server error")
        response = msgpack.unpackb(message, object_hook=msgpack_numpy.decode, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(response["error"])
        return response

    def wait_until_ready(self, *, timeout_s: float = 1800) -> None:
        """Block until the server answers a ping (it takes minutes to load a checkpoint)."""
        start = time.time()
        self._connect(5000)
        while time.time() - start < timeout_s:
            try:
                self.call(endpoint="ping")
                self._connect(self.timeout_ms)
                return
            except zmq.error.Again:
                self._connect(5000)
        raise TimeoutError(f"policy server at {self.host}:{self.port} not up after {timeout_s} s")

    def reset(self) -> None:
        self.call(endpoint="reset", data={"options": None})

    def get_action(self, *, observation: dict[str, Any]) -> dict[str, Any]:
        action, _ = self.call(endpoint="get_action", data={"observation": observation, "options": None})
        return action
