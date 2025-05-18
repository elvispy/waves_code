import jax.numpy as jnp
from collections.abc import Iterable

def simpson_weights(N: int, h: float) -> jnp.ndarray:
    """
    Returns Simpson's 1/3 rule weights for numerical integration on a uniform grid.

    Parameters:
    - N (int): Number of grid points (must be odd).
    - h (float): Spacing between grid points.

    Returns:
    - weights (jnp.ndarray): Vector of length N to be dot-multiplied with function values.
    """
    if N < 3 or N % 2 == 0:
        raise ValueError("N must be an odd integer greater than or equal to 3.")
    if isinstance(h, jnp.ndarray):
        # If not uniformly spaced, raise an error
        if jnp.max(jnp.diff(h)) != jnp.min(jnp.diff(h)):
            raise ValueError("Non uniform grids not implemented for integration.")
        h = (h[1] - h[0])  # Use the first spacing as the uniform spacing

    weights = jnp.ones(N)
    weights = weights.at[1:N-1:2].set(4.)
    weights = weights.at[2:N-2:2].set(2.)
    weights = weights * (h / 3.0)
    return weights
