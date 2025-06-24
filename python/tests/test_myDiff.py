"""test_myDiff.py
================================

Pytest suite exercising every JAX‑specific feature added in **`myDiff.py`**.

Run with::

    pytest -q test_myDiff.py

The suite is **self‑contained**: it stubs any missing `surferbot` utilities so
`myDiff` can be imported even outside the full Surferbot code‑base.
"""

from __future__ import annotations

# ───────────────────────── standard lib / test scaffolding ──────────────── #
import sys
import types
import importlib
from typing import Any

import numpy as np
import pytest

# ──────────────────────────── third‑party / JAX et al. ──────────────────── #
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

# ======================================================================== #
# 1.  Create **dummy** surferbot modules so that the wrapper imports OK
# ======================================================================== #

def _install_surferbot_stubs(sparse_default: bool = False) -> None:
    """Inject minimal stub modules under the *surferbot.* namespace."""
    # Parent *surferbot* package (namespace pkg is enough)
    if "surferbot" not in sys.modules:
        sys.modules["surferbot"] = types.ModuleType("surferbot")

    # Constants stub with a configurable *SPARSE* toggle
    const_mod = types.ModuleType("surferbot.constants")
    const_mod.SPARSE = sparse_default
    sys.modules["surferbot.constants"] = const_mod

    # sparse_utils stub – just needs the _SparseAtProxy class symbol
    sparse_utils_mod = types.ModuleType("surferbot.sparse_utils")

    class _SparseAtProxy:  # noqa: D401 – docstring not needed for stub
        """Ultra‑light proxy replicating the JAX ``x.at`` helper interface."""

        def __init__(self, *_args: Any, **_kw: Any):
            pass

        def _noop(self, *args: Any, **kwargs: Any):  # noqa: D401 – trivial stub
            return args[0] if args else None

        set = add = _noop  # type: ignore[assignment]

    sparse_utils_mod._SparseAtProxy = _SparseAtProxy  # type: ignore[attr-defined]
    sys.modules["surferbot.sparse_utils"] = sparse_utils_mod


# Ensure stubs are present *before* importing the wrapper for the first time
_install_surferbot_stubs()

# ───────────────────────────── import wrapper under test ────────────────── #
myDiff = importlib.import_module("myDiff")


# ========================================================================= #
# 2.  Helper fixtures
# ========================================================================= #

@pytest.fixture(params=[False, True], ids=["dense", "sparse"])
def wrapper(request):
    """Return the imported wrapper module with *SPARSE* toggled as requested."""
    myDiff.SPARSE = request.param  # flip global at runtime
    return myDiff


# ========================================================================= #
# 3.  Tests for **make_axis** and grid handling
# ========================================================================= #

GRID_CASES = [
    0.2,  # scalar spacing
    jnp.array([0.2]),  # one‑element JAX array spacing
    np.linspace(0.0, 1.0, 5),  # NumPy coordinate array
    jnp.linspace(0.0, 1.0, 5),  # JAX coordinate array
    jsparse.BCOO.fromdense(jnp.linspace(0.0, 1.0, 5)),  # sparse coordinates
]


@pytest.mark.parametrize("grid", GRID_CASES, ids=[
    "scalar‑spacing",
    "jax‑scalar‑spacing",
    "numpy‑coords",
    "jax‑coords",
    "bcoo‑coords",
])
def test_make_axis_accepts_all_grid_types(grid):
    """`Diff` should instantiate without error for every accepted grid spec."""
    diff = myDiff.Diff(axis=0, grid=grid, shape=(5,))
    assert diff is not None


# ========================================================================= #
# 4.  Tests for **Diff** core API additions
# ========================================================================= #

def test_scalar_left_multiplication_does_not_error(wrapper):
    """Expression ``1.0 * Diff`` should succeed and yield a JAX array/sparse."""
    X = wrapper.Diff(axis=0, grid=1.0, shape=(8,))
    out = 1.0 * X
    if wrapper.SPARSE:
        assert isinstance(out, jsparse.BCOO)
    else:
        assert isinstance(out, jnp.ndarray)


def test_op_dense_vs_sparse(wrapper):
    """`op()` must return dense *or* BCOO depending on the `SPARSE` flag."""
    Nx = 4
    diff = wrapper.Diff(axis=0, grid=1.0, shape=(Nx,))
    mat = diff.op()
    if wrapper.SPARSE:
        assert isinstance(mat, jsparse.BCOO)
    else:
        assert isinstance(mat, jnp.ndarray)
    assert mat.shape == (Nx, Nx)


def test_diff_power_increases_order():
    base = myDiff.Diff(axis=0, grid=1.0)
    higher = base ** 3
    assert higher.order == base.order * 3


def test_diff_mul_accumulates_order():
    d1 = myDiff.Diff(axis=0, grid=1.0)
    d2 = myDiff.Diff(axis=0, grid=1.0)
    combined = d1 * d2
    assert combined.order == d1.order + d2.order


def test_scalar_right_multiplication_returns_materialised_op(wrapper):
    d = wrapper.Diff(axis=0, grid=1.0, shape=(6,))
    out = d * 2.5
    if wrapper.SPARSE:
        assert isinstance(out, jsparse.BCOO)
    else:
        assert isinstance(out, jnp.ndarray)


def test_bcoo_has_at_property():
    mat = jsparse.BCOO.fromdense(jnp.eye(3))
    assert hasattr(mat, "at")
    assert hasattr(mat.at, "set")


# ========================================================================= #
# 5.  Tests for dynamic ``op`` patching after operator algebra
# ========================================================================= #

def test_composed_operator_has_jax_compatible_op(wrapper):
    d = wrapper.Diff(axis=0, grid=1.0, shape=(4,))
    vec = np.arange(4)
    composed = d * vec
    if composed is not NotImplemented:
        assert callable(getattr(composed, "op", None))
        mat = composed.op()
        if wrapper.SPARSE:
            assert isinstance(mat, jsparse.BCOO)
        else:
            assert isinstance(mat, jnp.ndarray)
