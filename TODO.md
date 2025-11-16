# TODO / Roadmap

## Testing (CPU-first)
- [ ] Add SRF component tests: semantic similarity, emotional weighting, associative strength, recency/decay, and SRF.retrieve deterministic checks.
- [ ] Extend cache/scheduler tests: insufficient blocks error handling, sequence removal/free, scheduler stats and cancel behavior.
- [ ] Add jarvis/serving stub tests: server create_app wiring, validate_config CLI, non-interactive flows beyond current dry-runs.
- [ ] Add coverage gating for CUDA tests (optional GPU job), keep CPU suite green by default.

## Typing
- [ ] Reduce mypy ignores: add annotations to `ai_assistant_pro/srf/components.py` and `engine/model.py`, then narrow ignore scopes.
- [ ] Add protocol/typed config objects for config/logging where reasonable.

## Kernels
- [ ] Stabilize FlashAttention Triton path (masking/softmax) to remove remaining xfail on WSL; retune block sizes if needed.
- [ ] Revisit FP8/quantization path (restore Triton-based FP8 or clearly document torch proxy).

## CI/Docs
- [ ] Add scripts/targets for CPU vs GPU test runs (export CUDA env vars), and optional pre-commit hook instructions.
- [ ] Re-enable GH CLI repo if needed; ensure apt sources are clean.
- [ ] Update README/CONTRIBUTING once tests/typing scope changes.
