# Contributing to ArchLens

Thank you for contributing to ArchLens! Please follow these guidelines to keep the codebase modular, privacy-first, and verifiable.

## Local Development Setup

```bash
git clone https://github.com/bhumibisen12/archlens.git
cd archlens
python -m venv .venv
# Activate your virtual environment, then install with dev tooling:
pip install -e ".[dev]"
pip install pre-commit
pre-commit install
```

Ensure the `tesseract` system binary is installed on your `PATH` if you run Tesseract integration tests locally.

## Code & Testing Standards

- **Python compatibility**: Write code compatible with Python 3.10 and 3.11 (CI runs on both).
- **One responsibility per module**: Keep functions small, type-annotated, and documented with concise docstrings.
- **Configuration over magic constants**: Place tunable pipeline thresholds in `src/archlens/config.py` and `configs/default.yaml`.
- **Privacy invariant**: Never introduce cloud AI APIs, runtime model downloads, external network calls, or telemetry.
- **Tests required**: Every module and bug fix must include `pytest` unit tests. Mock external binaries when testing unit behavior, and mark binary-dependent integration tests cleanly.
- **No fabricated metrics**: Only report metrics observed directly from running `scripts/evaluate.py` or `pytest`.

## Pre-PR Verification

Run the linter and full test suite before opening a pull request:

```bash
ruff check .
pytest -v
```

