from types import SimpleNamespace

import pytest

from app import daemon


def _settings(**overrides):
    base = {
        "visualizer_auto_start": True,
        "visualizer_bus_enabled": True,
        "visualizer_bus_host": "127.0.0.1",
        "visualizer_bus_port": 8765,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_visualizer_command_uses_bus_only_mode():
    cmd = daemon.build_visualizer_command(_settings())

    assert cmd[0].endswith("python")
    assert str(daemon.VISUALIZER_SCRIPT) in cmd
    assert "--no-mic" in cmd
    assert "--bus-host" in cmd
    assert "127.0.0.1" in cmd
    assert "--bus-port" in cmd
    assert "8765" in cmd


@pytest.mark.asyncio
async def test_start_visualizer_process_skips_when_autostart_disabled(monkeypatch):
    monkeypatch.setattr(
        daemon.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("should not launch visualizer"),
    )

    proc = await daemon.start_visualizer_process(
        _settings(visualizer_auto_start=False)
    )

    assert proc is None


@pytest.mark.asyncio
async def test_start_visualizer_process_skips_when_bus_disabled(monkeypatch):
    monkeypatch.setattr(
        daemon.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("should not launch visualizer"),
    )

    proc = await daemon.start_visualizer_process(
        _settings(visualizer_bus_enabled=False)
    )

    assert proc is None


def test_stop_visualizer_process_terminates_running_child():
    class FakeProc:
        returncode = None

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self) -> None:
            self.killed = True

    proc = FakeProc()

    daemon.stop_visualizer_process(proc)

    assert proc.terminated is True
    assert proc.killed is False


@pytest.mark.asyncio
async def test_start_visualizer_process_hides_pygame_banner(monkeypatch):
    captured = {}

    class FakeProc:
        pid = 123
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)

    proc = await daemon.start_visualizer_process(_settings())

    assert proc is not None
    assert captured["kwargs"]["env"]["PYGAME_HIDE_SUPPORT_PROMPT"] == "1"
    assert captured["kwargs"]["cwd"] == str(daemon.PROJECT_ROOT)
