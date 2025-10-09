# waves_code / surferbot

A collection of MATLAB and Python tools for building and testing hydrodynamic "surfer" models, including discrete DtN operators, mapped finite-difference utilities, and solver code for rigid and flexible surfboards/rafts.

This repository contains two primary work areas:
- `MATLAB/` — legacy and plotting scripts used for visualization and data generation.
- `python/` — a python package (`surferbot`) with numerical code (DtN, derivative operators, tests, and example solvers).

## Quick overview

The Python package implements utilities for constructing discrete Dirichlet-to-Neumann (DtN) operators, differentiation operators, and small solvers (e.g. `rigid_surferbot.py`) using JAX for array operations. The MATLAB folder contains plotting routines and legacy code used in experimentation.

## Minimal dev setup (macOS / zsh)

These instructions create a virtual environment, install the package in editable mode, and install runtime dependencies.

1. Create and activate a virtual environment from the `python/` folder:

```bash
cd /Users/eaguerov/Documents/Github/waves_code/python
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the package in editable mode:

```bash
pip install -e .
```

3. Install JAX (required by parts of the package). For CPU-only installs the upstream docs are the best source of truth; a common command is:

```bash
pip install --upgrade jax        # or follow https://github.com/google/jax#installation for pinned wheels / GPU support
```

Note: JAX wheels can vary by platform and accelerator (CPU/GPU) — follow the JAX README when in doubt.

## Running a simple example

After the editable install and dependencies are in place you can import or run modules from the package. Example (from repo root or anywhere with the venv activated):

```bash
# import and check DtN generator
python -c "from surferbot.DtN import DtN_generator; print('DtN OK, N=6 -> shape', DtN_generator(6).shape)"

# run the rigid surfer example module
python -m surferbot.rigid_surferbot
```

If you prefer not to install, export the package src directory to `PYTHONPATH` before running:

```bash
export PYTHONPATH="/Users/eaguerov/Documents/Github/waves_code/python/src:$PYTHONPATH"
python -m surferbot.rigid_surferbot
```

## Tests

The project contains pytest-based tests under `python/tests/`.

Run them with the venv active:

```bash
cd /Users/eaguerov/Documents/Github/waves_code/python
pytest -q
```

If tests fail with import errors, ensure the venv is active and that `pip install -e .` and dependencies (notably `jax`) are installed in that environment.

## Repository layout

- `MATLAB/` — MATLAB scripts and plotting utilities.
- `python/src/surferbot/` — Python package source code (DtN, derivative utilities, solvers).
- `python/tests/` — pytest test-suite for the Python package.

## Notes and troubleshooting

- If you see "No module named 'surferbot'", ensure you either installed the package with `pip install -e .` or exported `python/src` onto `PYTHONPATH`.
- If you see "No module named 'jax'" (or similar), install JAX in the active Python environment.
- On macOS the filesystem may be case-insensitive; still be careful to import using the correct capitalization (`from surferbot.DtN import ...`) as Python module names are case-sensitive at the language level.

## Contributing

Contributions and issues welcome. If you make changes to numerical routines please add or update tests under `python/tests/` and keep changes small and well-documented.

## License

This project uses the license declared in `python/pyproject.toml` (MIT by default). See repository files for exact terms.
