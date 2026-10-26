"""training-v1 telemetry (Sail guide section 10), rank 0 only: W&B summary fields refreshed every ~10 s by a background
thread, history under train/* and val/*. Without WANDB_API_KEY it only prints."""
import os
import threading
import time

PEAK_FLOPS = 2.25e15  # B300 dense bf16 per GPU (approximate)


class Telemetry:
    def __init__(self, *, project: str, total_steps: int, flops_per_step: float, n_gpus: int, config: dict | None = None):
        self.total_steps, self.flops_per_step, self.n_gpus = total_steps, flops_per_step, n_gpus
        self.step, self.step_time, self.mfu, self.url, self.run = 0, 0.0, 0.0, "", None
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb
                tag = os.environ.get("EXPERIMENT_TAG", project)
                self.run = wandb.init(project=project, id=os.environ.get("WANDB_RUN_ID", tag), name=tag,
                                      resume="allow", config=config or {})
                self.url = self.run.get_url()
            except Exception as e:  # telemetry must never kill training
                print(f"[telemetry] wandb init failed: {e!r}; continuing without W&B", flush=True)
                self.run = None
        else:
            print("[telemetry] WANDB_API_KEY not set: telemetry printed to the log only", flush=True)
        self._stop = False
        threading.Thread(target=self._beat, daemon=True).start()

    def _summary(self) -> dict:
        return {"pantheon_telemetry/contract_version": "training-v1",
                "pantheon_telemetry/heartbeat_at": int(time.time()),
                "pantheon_telemetry/total_steps": int(self.total_steps),
                "pantheon_telemetry/active_step": int(self.step),
                "pantheon_telemetry/step_time_s": float(self.step_time),
                "pantheon_telemetry/effective_mfu": float(min(max(self.mfu, 0.0), 1.0)),
                "pantheon_telemetry/wandb_url": self.url}

    def _beat(self):
        while not self._stop:
            if self.run is not None:
                try:
                    self.run.summary.update(self._summary())
                except Exception:
                    pass
            time.sleep(10)

    def update(self, *, step: int, step_time: float, logs: dict) -> None:
        """step_time: wall time of the step including data wait (end to end)."""
        self.step, self.step_time = step, step_time
        self.mfu = self.flops_per_step / max(step_time, 1e-6) / (PEAK_FLOPS * self.n_gpus)
        self.log(step=step, logs={**logs, "perf/effective_mfu": self.mfu, "perf/step_time_s": step_time})

    def log(self, *, step: int, logs: dict) -> None:
        if self.run is not None:
            try:
                self.run.log(logs, step=step)
            except Exception:
                pass

    def finish(self) -> None:
        self._stop = True
        if self.run is not None:
            try:
                self.run.summary.update(self._summary())
                self.run.finish()
            except Exception:
                pass
