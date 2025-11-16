# Repository Guidelines

## Project Structure & Modules
- Core package: `ai_assistant_pro/` with submodules for `srf/` (Stone Retrieval Function), `kernels/` (Triton + PyTorch kernels), `engine/` (model + scheduler + cache), and `serving/` (FastAPI server).
- Installer UX: `setup_jarvis.py` (GUI) and `install.py` (CLI); daemon/web entrypoints live under `ai_assistant_pro/jarvis`.
- Benchmarks in `benchmarks/`, runnable scripts under `examples/`, docs in `docs/`, and tests in `tests/`.
- Docker workflow: `Dockerfile` + `docker-compose.yml`; configuration template at `config.example.yaml`.

## Build, Test, Dev Commands
- Install dev deps: `pip install -e ".[dev]"`.
- Run locally (voice daemon): `ai-assistant-pro jarvis daemon --user-id "you"`.
- GUI installer: `python setup_jarvis.py`; CLI installer: `python install.py`.
- Lint/format: `black ai_assistant_pro/ examples/ benchmarks/`; `ruff check ai_assistant_pro/ examples/ benchmarks/`; `isort ai_assistant_pro/ tests/`; `flake8 ai_assistant_pro/ tests/`; optional `pre-commit run --all-files`.
- Type check: `mypy ai_assistant_pro/`.
- Tests: `pytest tests/`; add `-k <name>` to target a subset.
- CUDA-only tests require GPU and `RUN_CUDA_TESTS=1`.

## Coding Style & Naming
- Python 3.10+; Black + Ruff enforce style (100-char lines). Keep imports sorted with `isort` (black profile).
- Use type hints and Google-style docstrings; prefer explicit dataclasses/config objects over dicts.
- Kernels: place Triton code in `ai_assistant_pro/kernels/` with clear comments on SM120 optimizations.
- Naming: functions `lower_snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`; tests mirror module names (e.g., `test_srf.py`).

## Testing Guidelines
- Pytest-based. Co-locate fixtures in `tests/conftest.py` (add if needed); prefer fast unit tests over GPU-heavy runs by default.
- Cover new kernel paths with correctness checks plus minimal perf assertions when feasible; gate long benchmarks under markers.
- Name tests `test_<unit_behavior>` and keep deterministic seeds for stochastic components.

## Commit & PR Guidelines
- Commit messages: imperative subject with short body bullets for key changes (see history for examples: “Add …”, “Improve …”).
- Open PRs from feature branches; describe intent, surface benchmarks for performance work, and note hardware used (GPU/driver/CUDA).
- Link issues when applicable; include screenshots or logs for installer/UI changes and benchmark tables for kernel updates.

## Security & Configuration
- Do not commit secrets; keep local overrides in `config.example.yaml`-derived files ignored by Git.
- Validate incoming configs before use; prefer environment-variable overrides for deployment-sensitive settings.
