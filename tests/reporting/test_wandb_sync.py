import sys
import types

import pytest

from ignisca.reporting.wandb_sync import WandbSync


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.finished: bool = False

    def log(self, data: dict) -> None:
        self.logged.append(data)

    def finish(self) -> None:
        self.finished = True


def _install_fake_wandb(monkeypatch) -> tuple[list[dict], _FakeRun]:
    init_calls: list[dict] = []
    run = _FakeRun()

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return run

    fake_module = types.SimpleNamespace(init=fake_init)
    monkeypatch.setitem(sys.modules, "wandb", fake_module)
    return init_calls, run


def test_disabled_adapter_never_imports_wandb(monkeypatch):
    # Even if wandb is missing, a disabled adapter stays silent.
    monkeypatch.setitem(sys.modules, "wandb", None)  # force ImportError if touched
    sync = WandbSync(enabled=False, project="ignisca", run_name="cell_A1_seed0")
    sync.init_run()
    sync.log_eval({"iou": 0.6})
    sync.finish()  # no explosion


def test_enabled_adapter_calls_init_log_finish_in_order(monkeypatch):
    init_calls, run = _install_fake_wandb(monkeypatch)

    sync = WandbSync(enabled=True, project="ignisca", run_name="cell_A1_seed0")
    sync.init_run()
    assert len(init_calls) == 1
    assert init_calls[0]["project"] == "ignisca"
    assert init_calls[0]["name"] == "cell_A1_seed0"

    sync.log_eval({"iou": 0.61, "ece": 0.09})
    assert run.logged == [{"iou": 0.61, "ece": 0.09}]

    sync.finish()
    assert run.finished is True


def test_log_before_init_raises(monkeypatch):
    _install_fake_wandb(monkeypatch)
    sync = WandbSync(enabled=True, project="ignisca", run_name="cell_A1_seed0")
    with pytest.raises(RuntimeError, match="init_run"):
        sync.log_eval({"iou": 0.6})
