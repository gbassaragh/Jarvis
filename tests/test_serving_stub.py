import sys
import types

from click.testing import CliRunner

from ai_assistant_pro import cli as cli_module


def test_validate_config_cli_runs(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("engine:\n  model_name: tiny\n")

    # stub SRFConfig.validate to avoid pulling other deps
    class DummySRFConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def validate(self):
            return True

    dummy_pkg = types.ModuleType("ai_assistant_pro")
    dummy_pkg.__path__ = []
    dummy_srf = types.SimpleNamespace(SRFConfig=DummySRFConfig)
    monkeypatch.setitem(sys.modules, "ai_assistant_pro", dummy_pkg)
    monkeypatch.setitem(sys.modules, "ai_assistant_pro.srf", dummy_srf)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.validate_config,
        ["--output", str(tmp_path / "out.txt"), str(cfg)],
    )

    assert result.exit_code == 0
    assert "✓" in result.output


def test_serve_cli_start_server_stub(monkeypatch):
    captured = {}

    def fake_serve(**kwargs):
        captured.update(kwargs)

    dummy_pkg = types.ModuleType("ai_assistant_pro")
    dummy_pkg.__path__ = []
    serving_module = types.ModuleType("ai_assistant_pro.serving")
    serving_module.serve = fake_serve
    monkeypatch.setitem(sys.modules, "ai_assistant_pro", dummy_pkg)
    monkeypatch.setitem(sys.modules, "ai_assistant_pro.serving", serving_module)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["serve", "--host", "0.0.0.0", "--port", "9000", "--model", "stub", "--dry-run"],
    )

    assert result.exit_code == 0
    # dry-run should not call serve
    assert captured == {}
