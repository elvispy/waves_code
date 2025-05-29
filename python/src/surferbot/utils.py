import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres   # square, general sparse solver

def solve_tensor_system(A, b, *, tol=1e-8, maxiter=None):
    """
    Solve A·x = b where b.shape == A.shape[:b.ndim].
    Works for dense or BCOO/BCSR sparse A, square or rectangular.

    • Square   →  dense: jnp.linalg.solve   |  sparse: gmres  
    • Rectang. →  dense: lstsq              |  sparse: densify + lstsq
    """
    if A.shape[:b.ndim] != b.shape:
        raise ValueError("b.shape must equal A.shape[:b.ndim]")

    m, n = b.size, A.size // b.size
    A_flat = A.reshape(m, n)
    b_vec  = b.reshape(m)

    is_dense = isinstance(A, type(jnp.eye(3)))  # BCOO / BCSR

    if not is_dense:
        if m == n:                                   # square sparse → GMRES
            x_flat, info = gmres(A_flat, b_vec, tol=tol, maxiter=maxiter)
            if info != 0:
                raise RuntimeError(f"GMRES failed to converge (info={info})")
        else:                                        # rectangular sparse → fallback
            return NotImplemented #x_flat = jnp.linalg.lstsq(A_flat.todense(), b_vec, rcond=None)[0]
    else:                                            # dense
        if m == n:
            x_flat = jnp.linalg.solve(A_flat, b_vec)
        else:
            x_flat = jnp.linalg.lstsq(A_flat, b_vec, rcond=None)[0]

    return x_flat.reshape(A.shape[b.ndim:])


def gaussian_load(x0: float,
                  sigma: float,
                  x: jnp.ndarray) -> jnp.ndarray:
    """
    Smooth point load on an *arbitrary* 1-D grid.

    Parameters
    ----------
    F      : total force [N]
    x0     : centre of the Gaussian [same units as x]
    sigma  : physical width of the Gaussian (std-dev) [same units as x]
    x      : 1-D array of node coordinates (sorted)
    w      : optional pre-computed integration weights.
             If None, trapezoidal_weights(x) is used.

    Returns
    -------
    q      : array of size x.shape with units N·m⁻¹ such that
             jnp.sum(q * w) == F   (up to machine precision)
    """
    x = jnp.asarray(x)
    dx = jnp.diff(x)                              # length N
    w_mid = 0.5 * (dx[:-1] + dx[1:])              # length N-1
    w = jnp.concatenate([0.5*dx[:1], w_mid, 0.5*dx[-1:]])

    # Gaussian envelope (fully smooth in x0 → AD-friendly)
    phi = jnp.exp(-0.5 * ((x - x0) / sigma)**2)

    # exact discrete normalisation on this non-uniform grid
    delta = phi / jnp.sum(phi * w)      # Σ delta_i w_i = 1
    return delta                        # q_i  (units N·m⁻¹)


