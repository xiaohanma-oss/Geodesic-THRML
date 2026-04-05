# Contributing to Geodesic-THRML

Contributions are welcome! This document covers the essentials.

## Development setup

```bash
git clone https://github.com/xiaohanma-oss/Geodesic-THRML.git
cd Geodesic-THRML
pip install -e ".[dev]"          # installs jax, numpy, pln-thrml, pytest
```

## Running tests

```bash
pytest tests/ -v                 # all tests (~10 s)
```

## Code conventions

- **Docstrings**: all public functions should have a docstring with at least
  a one-line summary and a `Parameters` section for non-trivial signatures.
- **README sync**: if your change modifies the module structure, public API,
  or test file names, update `README.md` to match.

## Pull request process

1. Fork the repo and create a feature branch.
2. Make your changes — keep commits focused.
3. Run `pytest tests/ -v` and ensure all tests pass.
4. Open a PR against `main` with a short description of what and why.

## What to contribute

- Bridge implementations for new sub-projects
- Improved annealing schedules
- Evidence capsule merge strategies
- Hardware deployment experiments (TSU benchmarks)
- Documentation and examples
- Bug reports and fixes

## License

By contributing you agree that your contributions will be licensed under the
[MIT License](LICENSE).
