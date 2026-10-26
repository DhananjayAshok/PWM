"""training-v1 telemetry (cluster guide section 10): W&B summary fields refreshed every ~10 s by a heartbeat thread,
so the heartbeat stays fresh during cache building and validation too. Without WANDB_API_KEY it logs to stdout only."""
import os
import threading
import time
from typing import Any, Optional


class Telemetry:
    def __init__(self, *, project: str, run_id: str, total_steps: int, config: dict[str, Any]):
        self.total_steps = int(total_steps)
        self.active_step, self.step_time_s, self.effective_mfu = 0, 0.0, 0.0
        self.run = None
        if os.environ.get("WANDB_API_KEY"):
            import wandb
            self.run = wandb.init(project=project, id=run_id, name=run_id, resume="allow", config=config)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        while not self._stop.is_set():
            self.heartbeat()
            self._stop.wait(10.0)

    def heartbeat(self) -> None:
        if self.run is None:
            return
        self.run.summary.update({
            "pantheon_telemetry/contract_version": "training-v1",
            "pantheon_telemetry/heartbeat_at": time.time(),
            "pantheon_telemetry/total_steps": self.total_steps,
            "pantheon_telemetry/active_step": int(self.active_step),
            "pantheon_telemetry/step_time_s": float(self.step_time_s),
            "pantheon_telemetry/effective_mfu": float(min(max(self.effective_mfu, 0.0), 1.0)),
            "pantheon_telemetry/wandb_url": self.run.url,
        })

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def close(self, summary: Optional[dict[str, Any]] = None) -> None:
        self._stop.set()
        if self.run is not None:
            if summary:
                self.run.summary.update(summary)
            self.heartbeat()
            self.run.finish()
