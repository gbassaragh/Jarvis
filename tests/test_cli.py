import sys
import types

import pytest
from click.testing import CliRunner

from ai_assistant_pro import cli as cli_module


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help_lists_commands(runner):
    result = runner.invoke(cli_module.cli, ["--help"])

    assert result.exit_code == 0
    for command in ("generate", "serve", "benchmark", "validate-config", "jarvis"):
        assert command in result.output


def test_generate_invokes_engine(monkeypatch, runner):
    calls = {}

    class DummyEngine:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def generate(self, prompt, max_tokens, temperature):
            calls["generate"] = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            return "ok"

    dummy_pkg = types.ModuleType("ai_assistant_pro")
    dummy_pkg.AssistantEngine = DummyEngine
    monkeypatch.setitem(sys.modules, "ai_assistant_pro", dummy_pkg)

    result = runner.invoke(
        cli_module.generate,
        [
            "--model",
            "unit-test",
            "--prompt",
            "hello",
            "--max-tokens",
            "4",
            "--temperature",
            "0.5",
            "--no-triton",
            "--use-fp8",
        ],
    )

    assert result.exit_code == 0
    assert calls["init"]["model_name"] == "unit-test"
    assert calls["init"]["use_triton"] is False
    assert calls["init"]["use_fp8"] is True
    assert calls["generate"]["prompt"] == "hello"
    assert "Response" in result.output


def test_serve_invokes_server(monkeypatch, runner):
    captured = {}

    def fake_server(**kwargs):
        captured.update(kwargs)

    dummy_pkg = types.ModuleType("ai_assistant_pro")
    dummy_pkg.__path__ = []
    serving_module = types.ModuleType("ai_assistant_pro.serving")
    serving_module.serve = fake_server
    monkeypatch.setitem(sys.modules, "ai_assistant_pro", dummy_pkg)
    monkeypatch.setitem(sys.modules, "ai_assistant_pro.serving", serving_module)

    result = runner.invoke(
        cli_module.cli,
        ["serve", "--host", "127.0.0.1", "--port", "9999", "--model", "gpt2", "--no-triton"],
    )

    assert result.exit_code == 0
    assert captured == {
        "host": "127.0.0.1",
        "port": 9999,
        "model_name": "gpt2",
        "use_triton": False,
        "use_fp8": False,
    }


def test_validate_config_with_minimal_yaml(monkeypatch, runner, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "srf:\n  alpha: 0.4\n  beta: 0.2\nengine:\n  model_name: tiny\n"
    )

    validated = {}

    def fake_validate(self):
        validated["called"] = True

    from ai_assistant_pro import srf as srf_module

    monkeypatch.setattr(srf_module.SRFConfig, "validate", fake_validate)

    result = runner.invoke(
        cli_module.validate_config,
        ["--output", str(tmp_path / "out.txt"), str(config_path)],
    )

    assert result.exit_code == 0
    assert validated["called"] is True
    assert "✓" in result.output


def test_jarvis_chat_uses_stub(monkeypatch, runner):
    calls = {}

    class DummyJarvis:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

    dummy_pkg = types.ModuleType("ai_assistant_pro")
    dummy_pkg.__path__ = []
    dummy_module = types.SimpleNamespace(JARVIS=DummyJarvis)
    monkeypatch.setitem(sys.modules, "ai_assistant_pro", dummy_pkg)
    monkeypatch.setitem(sys.modules, "ai_assistant_pro.jarvis", dummy_module)

    result = runner.invoke(
        cli_module.cli,
        [
            "jarvis",
            "chat",
            "--model",
            "unit-test",
            "--user-id",
            "alice",
            "--no-tools",
            "--no-rag",
            "--no-memory",
            "--no-triton",
        ],
        input="quit\n",
    )

    assert result.exit_code == 0
    assert calls["init"]["model_name"] == "unit-test"
    assert calls["init"]["user_id"] == "alice"
    assert calls["init"]["enable_tools"] is False
    assert calls["init"]["enable_rag"] is False
    assert calls["init"]["enable_memory"] is False
    assert calls["init"]["use_triton"] is False


def test_srf_benchmark_uses_stubbed_benchmarks(monkeypatch, runner):
    calls = {"perf": 0, "quality": 0}

    class DummyPerf:
        def __init__(self, **kwargs):
            calls["perf"] += 1

        def run_all_benchmarks(self):
            calls["perf_run"] = True

    class DummyQuality:
        def __init__(self, **kwargs):
            calls["quality"] += 1

        def run_quality_benchmark(self):
            calls["quality_run"] = True

    dummy_bench_module = types.SimpleNamespace(
        SRFBenchmark=DummyPerf,
        SRFQualityBenchmark=DummyQuality,
    )

    monkeypatch.setitem(sys.modules, "benchmarks.srf_benchmark", dummy_bench_module)

    result = runner.invoke(
        cli_module.srf_benchmark,
        ["--n-candidates", "50", "--embedding-dim", "128"],
    )

    assert result.exit_code == 0
    assert calls["perf"] == 1
    assert calls["quality"] == 1
    assert calls.get("perf_run") is True
    assert calls.get("quality_run") is True
