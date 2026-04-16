from __future__ import annotations

from typing import Any, Optional


class WandbSync:
    """Opt-in Weights & Biases adapter.

    Inert when ``enabled=False``: ``init_run`` / ``log_eval`` / ``finish`` are
    no-ops and the ``wandb`` module is never imported. Tests monkeypatch
    ``sys.modules["wandb"]`` with a fake module to exercise the enabled path
    without touching the network.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        run_name: str,
        entity: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.project = project
        self.run_name = run_name
        self.entity = entity
        self._run: Any = None

    def init_run(self) -> None:
        if not self.enabled:
            return
        import wandb  # imported lazily — never during __init__

        kwargs: dict[str, Any] = {"project": self.project, "name": self.run_name}
        if self.entity is not None:
            kwargs["entity"] = self.entity
        self._run = wandb.init(**kwargs)

    def log_eval(self, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._run is None:
            raise RuntimeError("WandbSync.log_eval called before init_run")
        self._run.log(metrics)

    def finish(self) -> None:
        if not self.enabled:
            return
        if self._run is None:
            return
        self._run.finish()
        self._run = None
