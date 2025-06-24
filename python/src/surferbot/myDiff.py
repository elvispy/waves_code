"""surferbot.myDiff
===================
A thin, JAX‑compatible wrapper around ``findiff.Diff``.

The upstream *findiff* package builds sparse SciPy matrices or NumPy arrays,
which cannot participate in JAX transformations such as JIT or automatic
differentiation.  This module keeps the familiar ``Diff`` API while providing

* **Lazy materialisation** to either a dense ``jax.numpy`` array **or** a
  sparse ``jax.experimental.sparse.BCOO`` depending on the global
  :pydataattr:`surferbot.constants.SPARSE` flag.
* Full support for JAX *non‑equispaced* coordinate arrays and BCOO
  coordinates when building derivative stencils.
* Operator algebra (``d_dx * d_dz`` etc.) that still yields an object whose
  ``.op(shape)`` method returns a JAX‑native matrix.

Only comments and docstrings have been added – the *runtime behaviour is
unchanged*.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard / third‑party imports
# ---------------------------------------------------------------------------
from findiff import Diff as _Diff
from findiff import grids
import numpy as np
import numbers
import types

import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from surferbot.constants import SPARSE
from surferbot.sparse_utils import _SparseAtProxy

# Give JAX sparse arrays the same ``.at`` update helper as dense ones.
# We attach a *property* that returns a tiny proxy object replicating the API
#   x = jsparse.BCOO(...)
#   x = x.at[idx].set(value)
jsparse.BCOO.at = property(lambda self: _SparseAtProxy(self))

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_axis(dim: int, config_or_axis, periodic: bool = False):
    """Coerce *config_or_axis* into a *findiff* GridAxis.

    Parameters
    ----------
    dim
        Integer index of the dimension this axis represents.
    config_or_axis
        * One of the ``findiff.grids.*Axis`` instances – returned unchanged.
        * **Scalar** (``int`` or ``float``): uniform spacing.
        * **Length‑1 JAX array**: same as scalar, but convenient for traced
          code that produces an array scalar.
        * **NumPy / JAX ndarray or BCOO**: explicit, possibly non‑uniform
          coordinate list.
    periodic
        If *True*, produce a periodic finite‑difference stencil (wrap‑around
        boundary conditions).
    """
    # Already an axis → nothing to do
    if isinstance(config_or_axis, grids.GridAxis):
        return config_or_axis

    # Uniform spacing supplied as a plain Python/NumPy number
    if isinstance(config_or_axis, numbers.Number):
        return grids.EquidistantAxis(dim, spacing=float(config_or_axis),
                                     periodic=periodic)

    # Uniform spacing supplied as a *scalar* JAX array
    if isinstance(config_or_axis, jnp.ndarray) and config_or_axis.size == 1:
        # JAX regards shape ``(1,)`` as *not* scalar, so ``float(array)``
        # raises.  Extract the single element explicitly.
        spacing = float(jnp.reshape(config_or_axis, ()).item())  # → Python scalar
        return grids.EquidistantAxis(dim, spacing=spacing, periodic=periodic)

    # Explicit coordinate list – any dense or sparse array type works here
    if isinstance(config_or_axis, (np.ndarray, jnp.ndarray, jsparse.BCOO)):
        return grids.NonEquidistantAxis(dim, coords=config_or_axis,
                                        periodic=periodic)

    # Anything else is a bug in the caller
    raise TypeError(f"Unsupported grid specification for axis {dim}: "
                    f"{type(config_or_axis).__name__}")


def _dynamic_s_op_method(S: _Diff, shape: tuple[int, ...] | None = None):
    """Materialise **S** into a JAX array or BCOO with broadcasting shape.

    The upstream ``findiff.Diff.matrix`` method yields a *SciPy* sparse
    matrix.  We convert that to either dense ``jax.numpy`` or sparse BCOO and
    then reshape it to ``shape + shape`` so that it acts on an *N‑D field* via
    einsum‑style contraction.
    """
    target_shape = shape if shape is not None else getattr(S, "shape", None)

    # We only support 1‑D or 2‑D axes out of laziness; higher‑rank reshaping
    # would need a full kronecker product.  Returning *NotImplemented* lets
    # users fall back on ``S.matrix`` if they really need that.
    if not (isinstance(target_shape, tuple) and len(target_shape) <= 2):
        return NotImplemented

    # Build the SciPy matrix once…
    scipy_mat = S.matrix(target_shape)

    if SPARSE:
        # … convert to JAX sparse (still CSR under the hood) …
        jax_mat = jsparse.BCOO.from_scipy_sparse(scipy_mat)
    else:
        # … or dense JAX array.
        jax_mat = jnp.asarray(scipy_mat.toarray())

    # Finally reshape to a 4‑D tensor that maps (rows, cols)
    return jax_mat.reshape(*(target_shape + target_shape))

# ---------------------------------------------------------------------------
# Derivative operator class
# ---------------------------------------------------------------------------

class Diff(_Diff):
    """Drop‑in replacement for *findiff* ``Diff`` with JAX semantics.

    Parameters
    ----------
    axis, grid, periodic, acc
        Same meaning as in upstream *findiff* plus the extra *periodic*
        convenience on the wrapper side.
    shape
        Expected shape of the array this operator will act on.  Needed so that
        :py:meth:`op` can lazily build a correctly‑shaped tensor.
    """

    # The signature mirrors the original but adds *shape* and *periodic* in a
    # non‑keyword‑only position so existing user code keeps working.
    def __init__(self, axis: int = 0, grid=None, shape: tuple[int, ...] | None = None,
                 periodic: bool = False, acc: int = _Diff.DEFAULT_ACC):
        # Convert spacing / coords → GridAxis, *then* call the parent ctor.
        grid_axis = make_axis(axis, grid, periodic)
        super().__init__(axis, grid_axis, acc)

        # Remember the target field shape for on‑demand materialisation
        self.shape = shape

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def op(self, shape: tuple[int, ...] | None = None):
        """Return a JAX matrix (dense or BCOO) acting on ``shape``.

        If *shape* is **None** we fall back to the ``shape`` attribute provided
        at construction time.
        """
        return _dynamic_s_op_method(self, shape)

    # ------------------------------------------------------------------
    # Operator algebra (++, ** etc.) – we just need to propagate *shape*
    # ------------------------------------------------------------------

    def __pow__(self, power: int):
        """Return a *new* ``Diff`` representing a higher‑order derivative."""
        new = Diff(self.dim, self.axis, shape=self.shape, acc=self.acc)
        new._order *= power
        return new

    def __mul__(self, other):
        """Chain operators or apply numeric left‑multiplication."""

        # (1) Diff * Diff  → higher‑order derivative (dim must match)
        if isinstance(other, Diff) and self.dim == other.dim:
            new = Diff(self.dim, self.axis, shape=self.shape, acc=self.acc)
            new._order += other.order
            return new

        # (2) scalar / JAX array * Diff  → materialise matrix and multiply
        if isinstance(other, (numbers.Number, jnp.ndarray)):
            return other * self.op()

        # (3) Fallback to upstream behaviour, then retrofit JAX semantics if
        #     the two operands share a *.shape* attribute (i.e. came from this
        #     wrapper).  That gives us a ready‑to‑use ``.op`` again.
        S = super().__mul__(other)
        if getattr(self, "shape", None) == getattr(other, "shape", None):
            S.shape = self.shape
            S.op = types.MethodType(_dynamic_s_op_method, S)
            return S

        return NotImplemented

    # Right‑hand scalar multiplication delegates to __mul__ for brevity
    def __rmul__(self, other):
        if isinstance(other, (numbers.Number, jnp.ndarray)):
            return self.__mul__(other)
        return super().__rmul__(other)
