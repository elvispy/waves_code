"""test_myDiff.py
================================

Pytest suite exercising every JAX‑specific feature added in **`myDiff.py`**.

Run with::

    pytest -q python/tests/test_myDiff.py

This file must live inside *python/tests/*.  It bootstraps a couple of stub
modules (``surferbot.constants`` and ``surferbot.sparse_utils``) **only if they
aren’t already importable** – so the rest of your code‑base keeps working.  The
stubs now expose both ``SPARSE`` *and* ``DEBUG`` flags because other test files
(e.g. *test_solver.py*) import ``surferbot.constants.DEBUG`` during collection.
"""

from __future__ import annotations

# ───────────────────────── standard lib / test scaffolding ──────────────── #
import sys
import types
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ──────────────────────────── third‑party / JAX et al. ──────────────────── #
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

# ======================================================================== #
# 0.  Make sure *project* root (containing python/src) is on sys.path
# ======================================================================== #
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # → <repo>/
SRC_DIR = PROJECT_ROOT / "python" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ======================================================================== #
# 1.  Create **dummy** surferbot modules (only if missing)
# ======================================================================== #

def _install_surferbot_stubs(sparse_default: bool = False) -> None:  # noqa: D401
    """Inject minimal stub modules under the *surferbot.* namespace.

    We *only* create a stub if the real module cannot be imported.  This lets
    developers replace the stub by adding the actual package to PYTHONPATH.
    """

    try:
        importlib.import_module("surferbot.constants")  # type: ignore[import-not-found]
        return  # Real package is available – nothing to patch
    except ModuleNotFoundError:
        pass

    # Parent *surferbot* namespace package
    sys.modules.setdefault("surferbot", types.ModuleType("surferbot"))

    # ── surferbot.constants ──────────────────────────────────────────────── #
    const_mod = types.ModuleType("surferbot.constants")
    const_mod.SPARSE = sparse_default
    const_mod.DEBUG = False  # <- added so other legacy tests can import DEBUG
    sys.modules["surferbot.constants"] = const_mod

    # ── surferbot.sparse_utils ───────────────────────────────────────────── #
    sparse_utils_mod = types.ModuleType("surferbot.sparse_utils")

    class _SparseAtProxy:  # noqa: D401 – stub only
        """Very small proxy replicating JAX ``x.at`` API for sparse BCOO."""

        def __init__(self, *_args: Any, **_kw: Any):
            pass

        def _noop(self, *args: Any, **kwargs: Any):  # noqa: D401 – noop
            return args[0] if args else None

        set = add = _noop  # type: ignore[assignment]

    sparse_utils_mod._SparseAtProxy = _SparseAtProxy  # type: ignore[attr-defined]
    sys.modules["surferbot.sparse_utils"] = sparse_utils_mod


_install_surferbot_stubs()

# ───────────────────────────── import wrapper under test ────────────────── #
# The wrapper lives at python/src/surferbot/myDiff.py → import as a sub‑module
myDiff = importlib.import_module("surferbot.myDiff")

# ========================================================================= #
# 2.  Helper fixtures
# ========================================================================= #

@pytest.fixture(params=[False, True], ids=["dense", "sparse"])
def wrapper(request):
    """Return the imported wrapper module with *SPARSE* toggled as requested."""
    myDiff.SPARSE = bool(request.param)  # flip global at runtime
    return myDiff

# ========================================================================= #
# 3.  Tests for **make_axis** and grid handling
# ========================================================================= #

GRID_CASES = [
    0.2,  # scalar spacing
    jnp.array([0.2]),  # 1‑element JAX array spacing
    np.linspace(0.0, 1.0, 5),  # NumPy coordinate array
    jnp.linspace(0.0, 1.0, 5),  # JAX coordinate array
    jsparse.BCOO.fromdense(jnp.linspace(0.0, 1.0, 5)),  # sparse coords
]


@pytest.mark.parametrize("grid", GRID_CASES, ids=[
    "scalar‑spacing",
    "jax‑scalar‑spacing",
    "numpy‑coords",
    "jax‑coords",
    "bcoo‑coords",
])
def test_make_axis_accepts_all_grid_types(grid):
    diff = myDiff.Diff(axis=0, grid=grid, shape=(5,))
    assert diff  # construction succeeded

# ========================================================================= #
# 4.  Tests for **Diff** core API additions
# ========================================================================= #

def test_scalar_left_multiplication_does_not_error(wrapper):
    X = wrapper.Diff(axis=0, grid=1.0, shape=(8,))
    out = 1.0 * X
    assert isinstance(out, (jsparse.BCOO if wrapper.SPARSE else jnp.ndarray))


def test_op_dense_vs_sparse(wrapper):
    Nx = 4
    diff = wrapper.Diff(axis=0, grid=1.0, shape=(Nx,))
    mat = diff.op()
    assert isinstance(mat, (jsparse.BCOO if wrapper.SPARSE else jnp.ndarray))
    assert mat.shape == (Nx, Nx)


def test_diff_power_increases_order():
    assert (myDiff.Diff(axis=0, grid=1.0) ** 3).order == myDiff.Diff(axis=0, grid=1.0).order * 3


def test_diff_mul_accumulates_order():
    d1 = myDiff.Diff(axis=0, grid=1.0)
    d2 = myDiff.Diff(axis=0, grid=1.0)
    assert (d1 * d2).order == d1.order + d2.order


def test_scalar_right_multiplication_returns_materialised_op(wrapper):
    d = wrapper.Diff(axis=0, grid=1.0, shape=(6,))
    out = d * 2.5
    assert isinstance(out, (jsparse.BCOO if wrapper.SPARSE else jnp.ndarray))


def test_bcoo_has_at_property():
    mat = jsparse.BCOO.fromdense(jnp.eye(3))
    assert hasattr(mat, "at") and hasattr(mat.at, "set")

# ========================================================================= #
# 5.  Tests for dynamic ``op`` patching after operator algebra
# ========================================================================= #

def test_composed_operator_has_jax_compatible_op(wrapper):
    d = wrapper.Diff(axis=0, grid=1.0, shape=(4,))
    vec = np.arange(4)
    composed = d * vec
    if composed is not NotImplemented:
        mat = composed.op()
        assert isinstance(mat, (jsparse.BCOO if wrapper.SPARSE else jnp.ndarray))
