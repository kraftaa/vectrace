# PyPI Release Guide

This document is the release checklist for publishing `vectrace` to PyPI.

## 0) GitHub Workflow Location

Workflow files are in:
- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`

`release.yml` publishes to PyPI when a GitHub Release is published (or by manual run).

## 1) Preflight

- Update version in `pyproject.toml`.
- Set real `[project.urls]` in `pyproject.toml` (Homepage/Repository/Issues) before release.
- Run tests:
  - `python3 -m unittest discover -s tests -v`
  - `.venv/bin/python -m unittest discover -s tests -v`
- Ensure `README.md`, `LICENSE`, and release notes are up to date.

## 2) Build Artifacts

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install build twine setuptools wheel
python3 -m build
python3 -m twine check dist/*
```

Expected outputs:
- `dist/vectrace-<version>.tar.gz`
- `dist/vectrace-<version>-py3-none-any.whl`

## 3) Upload

TestPyPI first:

```bash
python3 -m twine upload --repository testpypi dist/*
```

Then production PyPI:

```bash
python3 -m twine upload dist/*
```

If you use GitHub Actions trusted publishing, configure PyPI trusted publisher for this repo and rely on `.github/workflows/release.yml` instead of manual `twine upload`.

## 4) Verify Install

```bash
python3 -m venv /tmp/vectrace-release-test
source /tmp/vectrace-release-test/bin/activate
python3 -m pip install vectrace
vectrace --help
vectrace onboard --db /tmp/vectrace-release.db --output /tmp/vectrace-release.html
```

## 5) Post-Release

- Tag release in git (`v<version>`).
- Publish release notes.
- Update launch channels (X/LinkedIn/Reddit/HN) with install command and 30s demo.
