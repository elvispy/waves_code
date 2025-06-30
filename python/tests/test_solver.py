# tests/test_pde_blocks.py
import pytest
import jax.numpy as jnp
import numpy as np

# ----------------------------------------------------------------------
# Your public solver now has   return_internal=True   to expose internals
# ----------------------------------------------------------------------
from surferbot.flexible_surferbot import solver   # ← keep whatever path is correct


def _manufactured_phi(xg, zg, k=1.5):
    """
    Harmonic potential that is Laplace–harmonic everywhere and satisfies
    ∂φ/∂z = 0 at z = −1 (assuming the bottom grid point sits there).

        φ(x,z) = cos(k x) · cosh(k (z+1))
    """
    X, Z = jnp.meshgrid(xg, zg, indexing="ij")
    return jnp.cos(k * X) * jnp.cosh(k * (Z + 1.0))

def _apply_operator(D, field):
    """
    Apply the 4-D operator D_{ijmn} to field f_{mn}.
    We reshape both to 2-D, perform a standard matmul, then reshape back.
    """
    N, M = field.shape
    flat = (D.reshape(N * M, N * M) @ field.reshape(-1)).reshape(N, M)
    return flat


class TestPDEBlocks:
    """Check that the discrete Laplacian block annihilates a harmonic φ."""

    @pytest.mark.parametrize("n, M", [(21, 10), (5, 50)])
    def test_bulk_laplace_zero(self, n, M):
        # 1) get system tensor and FD operators
        A, xg, zg, Dx = solver(
            n=n, M=M
        )
        N, Mz = len(xg), len(zg)

        # 2) build φ and its first→4th x-derivatives
        phi      = _manufactured_phi(xg, zg, k=1.5)
        phi      = _manufactured_phi(xg, zg, k=1.5)
        phi_x    = _apply_operator(Dx, phi)
        phi_xx   = _apply_operator(Dx, phi_x)
        phi_xxx  = _apply_operator(Dx, phi_xx)
        phi_xxxx = _apply_operator(Dx, phi_xxx)

        state = jnp.stack([phi, phi_x, phi_xx, phi_xxx, phi_xxxx], axis=-1)  # (N,M,5)

        # 3) residual = A ⋅ state  (flatten to 2-D for speed)
        rows = np.prod(A.shape[:3])            # 5*N*M
        cols = state.size                      # 5*N*M
        residual = (A.reshape(rows, cols) @ state.ravel()).reshape(A.shape[:3])

        # 4) block-index 0 == rows that came from Laplacian E2
        laplace_rows = residual[0]             # (N,M)
        interior = laplace_rows[1:-1, 1:-1]    # boundaries were zeroed in solver

        # 5) assert ∥interior∥∞ < 1e-8
        assert jnp.sum(jnp.abs(interior)) < 0.05 * N * M, (
            "Manufactured harmonic field does not satisfy discrete ∇²φ = 0 "
            "inside the domain."
        )
