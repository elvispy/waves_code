import jax.numpy as jnp
from typing import Union

def simpson_weights(
    N: int,
    h: Union[float, jnp.ndarray]
) -> jnp.ndarray:
    """
    Simpson's 1/3 rule weights on either a uniform or a non-uniform grid.

    Parameters
    ----------
    N : int
      Number of grid points (must be odd and >= 3).
      If `h` is an array, N is ignored and inferred from len(h).
    h : float or jnp.ndarray
      - If float: uniform spacing.
      - If 1-D array of length N: the grid coordinates (must be sorted).

    Returns
    -------
    w : jnp.ndarray
      Weights such that `sum(w * f)` approximates ∫ f over the grid.
    """
    # Non-uniform grid branch
    if isinstance(h, jnp.ndarray):
        x = h
        N = x.shape[0]
        if N < 3 or N % 2 == 0:
            raise ValueError("Grid length must be odd and >= 3 for Simpson’s rule.")
        w = jnp.zeros(N, dtype=x.dtype)

        # Composite Simpson: loop over each pair of intervals [x[i], x[i+1], x[i+2]]
        for i in range(0, N - 2, 2):
            x0, x1, x2 = x[i], x[i+1], x[i+2]
            h0 = x1 - x0
            h1 = x2 - x1

            # Exact quadratic-interpolation weights:
            w0 =   h0/3 +  h1/6   - (h1**2)/(6*h0)
            w1 =  (h0**2)/(6*h1) + h0/2 + h1/2 + (h1**2)/(6*h0)
            w2 =  -(h0**2)/(6*h1) + h0/6 + h1/3

            w = w.at[i  ].add(w0)
            w = w.at[i+1].add(w1)
            w = w.at[i+2].add(w2)

        return w

    # Uniform grid branch
    else:
        if N < 3 or N % 2 == 0:
            raise ValueError("N must be an odd integer >= 3 for Simpson’s rule.")
        weights = jnp.ones(N)
        weights = weights.at[1:N-1:2].set(4.0)
        weights = weights.at[2:N-2:2].set(2.0)
        return weights * (h / 3.0)
