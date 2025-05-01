import jax.numpy as jnp

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

    weights = jnp.ones(N)
    weights = weights.at[1:N-1:2].set(4)
    weights = weights.at[2:N-2:2].set(2)
    weights = weights * (h / 3.0)
    return weights
