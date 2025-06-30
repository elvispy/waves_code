"""pytest suite for surferbot.myDiff

Covers every behaviour added by the thin wrapper:
* accepts scalar spacing and explicit coordinate arrays (NumPy / JAX / BCOO)
* `.op(shape)` yields a dense or sparse *JAX* matrix, not SciPy
* operator algebra (power, left/right multiplication) preserves wrapper
  semantics
* new regression: 2‑D shape handling and scalar‑multiplication materialisation
"""

from __future__ import annotations

import importlib
import types
from copy import deepcopy

import pytest
import numpy as np
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

# ---------------------------------------------------------------------------
# Dynamically import the wrapper under test
# ---------------------------------------------------------------------------
myDiff = importlib.import_module("surferbot.myDiff")

# ---------------------------------------------------------------------------
# Helpers to build sample grids
# ---------------------------------------------------------------------------

Nx = 5
COORDS_NUMPY = np.linspace(0.0, 1.0, Nx, dtype=float)
COORDS_JAX   = jnp.linspace(0.0, 1.0, Nx)

# 1‑D coordinate array stored sparsely (just to hit that code path)
data   = COORDS_NUMPY.astype(np.float32)
indices = np.arange(Nx)[:, None]  # each coord sits in its own row of a (Nx,1)
COORDS_BCOO = jsparse.BCOO((data, indices), shape=(Nx,))

# ---------------------------------------------------------------------------
# Parametrised cases for make_axis
# ---------------------------------------------------------------------------

GRID_CASES = [
    0.2,                            # scalar spacing
    jnp.array([0.2], dtype=jnp.float32),  # 1‑element JAX array (regression!)
    COORDS_NUMPY,                   # explicit NumPy coords
    COORDS_JAX,                     # explicit JAX coords
    COORDS_BCOO,                    # explicit BCOO coords
]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid", GRID_CASES, ids=[
    "scalar‑spacing",
    "jax‑scalar‑spacing",
    "numpy‑coords",
    "jax‑coords",
    "bcoo‑coords",
])
def test_make_axis_accepts_all_grid_types(grid):
    """Construction should never raise for any supported grid description."""
    diff = myDiff.Diff(axis=0, grid=grid, shape=(Nx,))
    assert isinstance(diff, myDiff.Diff)


def test_op_dense_vs_sparse(monkeypatch):
    """`SPARSE` flag controls whether `.op` yields BCOO or dense array."""
    Nx = 4
    diff = myDiff.Diff(axis=0, grid=1.0, shape=(Nx,))

    # Force dense
    monkeypatch.setattr(myDiff, "SPARSE", False, raising=False)
    dense = diff.op()
    assert isinstance(dense, jnp.ndarray) and dense.shape == (Nx, Nx)

    # Force sparse
    monkeypatch.setattr(myDiff, "SPARSE", True, raising=False)
    sparse = diff.op()
    assert isinstance(sparse, jsparse.BCOO) and sparse.shape == (Nx, Nx)


def test_pow_and_mul_order():
    """`**` and `*` on two Diff objects should add / multiply derivative order."""
    d = myDiff.Diff(axis=0, grid=1.0, shape=(Nx,))
    d2 = d ** 2
    assert d2.order == 2 * d.order

    dd = d * d  # same axis ⇒ order add
    assert dd.order == 2 * d.order


def test_rmul_scalar_produces_matrix():
    """Left‑scaling by a scalar materialises the operator tensor."""
    Nx, Mx = 3, 2
    diff = myDiff.Diff(axis=0, grid=1.0, shape=(Nx, Mx))
    tensor = 1.0 * diff  # triggers __rmul__
    assert tensor.shape == (Nx, Mx, Nx, Mx)


# ---------------------------------------------------------------------------
# NEW regression: shape=(N, M) with coordinate grid (JAX array)
# ---------------------------------------------------------------------------

def test_op_matrix_shape_2d():
    """`.op` (via scalar‑multiplication) for 2‑D shape returns (N,M,N,M) tensor."""
    N, M = 4, 3
    grid_x = jnp.linspace(0.0, 1.0, N)
    diff = myDiff.Diff(axis=0, grid=grid_x, shape=(N, M))

    mat = 1.0 * diff  # should hit __rmul__ → .op → JAX array/BCOO
    assert mat.shape == (N, M, N, M)


