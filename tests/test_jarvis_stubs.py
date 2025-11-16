import sys
import types

from click.testing import CliRunner

from ai_assistant_pro import cli as cli_module


def test_jarvis_chat_cli_dry_run(monkeypatch):
    calls = {}

    class DummyJarvis:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

    dummy_pkg = types.ModuleType("ai_assistant_pro")
    dummy_pkg.__path__ = []
    dummy_jarvis = types.SimpleNamespace(JARVIS=DummyJarvis)
    monkeypatch.setitem(sys.modules, "ai_assistant_pro", dummy_pkg)
    monkeypatch.setitem(sys.modules, "ai_assistant_pro.jarvis", dummy_jarvis)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "jarvis",
            "chat",
            "--model",
            "cpu-stub",
            "--user-id",
            "tester",
            "--no-tools",
            "--no-rag",
            "--no-memory",
            "--no-triton",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert calls["init"]["model_name"] == "cpu-stub"
    assert calls["init"]["user_id"] == "tester"
    assert calls["init"]["enable_tools"] is False
    assert calls["init"]["enable_rag"] is False
    assert calls["init"]["enable_memory"] is False
    assert calls["init"]["use_triton"] is False


def test_serve_cli_dry_run(monkeypatch):
    captured = {}

    def fake_server(**kwargs):
        captured.update(kwargs)

    dummy_pkg = types.ModuleType("ai_assistant_pro")
    dummy_pkg.__path__ = []
    serving_module = types.ModuleType("ai_assistant_pro.serving")
    serving_module.serve = fake_server
    monkeypatch.setitem(sys.modules, "ai_assistant_pro", dummy_pkg)
    monkeypatch.setitem(sys.modules, "ai_assistant_pro.serving", serving_module)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["serve", "--host", "127.0.0.1", "--port", "8080", "--model", "cpu-stub", "--dry-run"],
    )

    assert result.exit_code == 0
    assert captured == {}  # dry run should not invoke server
