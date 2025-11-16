# Contributing to AI Assistant Pro

Thank you for your interest in contributing to AI Assistant Pro! This document provides guidelines for contributions.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-assistant-pro.git
   cd ai-assistant-pro
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Guidelines

### Code Style

We use:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking
- **isort** for import ordering (black profile)
- **Flake8** for additional linting

Run formatters:
```bash
black ai_assistant_pro/ examples/ benchmarks/
ruff check ai_assistant_pro/ examples/ benchmarks/
mypy ai_assistant_pro/
isort ai_assistant_pro/ tests/
flake8 ai_assistant_pro/ tests/
pre-commit run --all-files
```

### Testing

Run tests:
```bash
pytest tests/
```

Add tests for new features in `tests/`.

CUDA-only kernel tests are gated; set `RUN_CUDA_TESTS=1` and ensure a GPU is available to run them.

### Custom Kernels

When adding new Triton kernels:
1. Add kernel implementation in `ai_assistant_pro/kernels/`
2. Include docstrings with:
   - Description of the operation
   - SM120-specific optimizations
   - Performance characteristics
3. Add Python wrapper function
4. Add benchmarks in `benchmarks/`
5. Add examples in `examples/`

### Performance

All new kernels should:
- Be faster than PyTorch baseline
- Include memory usage analysis
- Have comprehensive benchmarks
- Document expected speedups

### Documentation

- Use Google-style docstrings
- Include type hints
- Add usage examples
- Update README.md if needed

## Commit Guidelines

Use descriptive commit messages:
```
Add FlashAttention-4 kernel for SM120

- Implement warp-specialized attention
- Add FP8 support for 2x speedup
- Include comprehensive benchmarks
```

## Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub

5. Ensure CI passes:
   - Tests pass
   - Code style checks pass
   - Documentation builds

## Areas for Contribution

We welcome contributions in:

### High Priority
- **Additional Triton kernels** (grouped query attention, sparse attention)
- **Quantization methods** (INT4, mixed precision)
- **Model support** (LLaMA 3, Mixtral, etc.)
- **Benchmarks** (comparison with other frameworks)

### Medium Priority
- **Multi-GPU** (tensor parallelism, pipeline parallelism)
- **Speculative decoding** (draft model integration)
- **Continuous batching** (advanced scheduling algorithms)

### Documentation
- **Tutorials** (step-by-step guides)
- **Examples** (real-world use cases)
- **Performance guides** (tuning for specific hardware)

## Hardware Requirements

Development requires:
- NVIDIA GPU (SM90+ recommended, SM120 for full features)
- CUDA 12.4+
- 16GB+ GPU memory (for testing large models)

## Questions?

Open an issue or discussion on GitHub!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
