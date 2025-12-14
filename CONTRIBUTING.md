# Contributing

Thanks for your interest in improving **ASL&AI**.

## Development setup

```bash
make setup
source env/bin/activate
make install
```

## Recommended workflow

```bash
make format
make lint
make test
```

## Pre-commit (recommended)

```bash
python -m pip install -r requirements-dev.txt
python -m pip install pre-commit
pre-commit install
```

## Pull requests

- Keep PRs focused and small.
- Add/adjust tests for behavior changes.
- Update docs (`README.md`, `QUICKSTART.md`) when user-facing behavior changes.


