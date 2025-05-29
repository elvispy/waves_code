import jax
import jax.numpy as jnp

# JAX is typically run on CPU by default if no GPU/TPU is found.
# You can explicitly set the default device if needed, e.g.:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])

def calculate_y_exponential_jax(x_val: jax.Array,
                                x_i_array: jax.Array,
                                F_val: jax.Array,
                                k_val: jax.Array) -> jax.Array:
    """
    Calculates the distributed values y_i based on the exponential decay (Gaussian kernel) method.

    This function is JAX-compatible and supports automatic differentiation.

    Args:
        x_val: The input value x (scalar).
        x_i_array: A 1D JAX array of values [x1, x2, ..., xn].
        F_val: The total sum F that the y_i values should sum to (scalar).
        k_val: The positive constant k controlling the width of the influence (scalar).
               A larger k means a narrower, more focused peak.

    Returns:
        A 1D JAX array [y1, y2, ..., yn].
    """
    # Ensure inputs are JAX arrays (though JAX often handles conversion)
    # For clarity and to ensure they are treated as JAX arrays for AD:
    x = jnp.asarray(x_val, dtype=jnp.float32)
    xis = jnp.asarray(x_i_array, dtype=jnp.float32)
    F = jnp.asarray(F_val, dtype=jnp.float32)
    k = jnp.asarray(k_val, dtype=jnp.float32)

    # Step 1: Calculate squared differences (x - xi)^2
    # (x - xi)
    diffs = x - xis  # Broadcasting x with each element of xis
    # (x - xi)^2
    squared_diffs = diffs**2

    # Step 2: Calculate raw scores si = exp(-k * (x - xi)^2)
    # -k * (x - xi)^2
    exponents = -k * squared_diffs
    # exp(-k * (x - xi)^2)
    raw_scores_s_i = jnp.exp(exponents)

    # Step 3: Calculate the sum of raw scores S = sum(si)
    sum_of_raw_scores_S = jnp.sum(raw_scores_s_i)

    # Prevent division by zero if all raw scores are zero (e.g., k is extremely large and x is far from all xis)
    # Add a small epsilon to the denominator for numerical stability,
    # though with exp, scores are always positive unless exponents are -inf.
    # A more robust check might be if sum_of_raw_scores_S is very close to zero.
    # For simplicity, we assume F > 0 and at least one s_i > 0 (which exp ensures unless k is inf or exponent is -inf).
    # If sum_of_raw_scores_S is zero, this would lead to NaNs or Infs.
    # JAX handles this by propagating NaNs, which is often desired in AD.
    # If sum_of_raw_scores_S could legitimately be zero and you need a different behavior (e.g. uniform distribution),
    # you might add a jnp.where condition here.

    # Step 4: Calculate yi = F * (si / S)
    y_i_array = F * (raw_scores_s_i / sum_of_raw_scores_S)

    return y_i_array

if __name__ == '__main__':
    # Example Usage:

    # Define some input values
    x_example = jnp.array(2.5, dtype=jnp.float32)
    xi_example = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    F_example = jnp.array(100.0, dtype=jnp.float32)
    k_example = jnp.array(1.0, dtype=jnp.float32) # Controls the "spread"

    # Calculate y_i values
    y_values = calculate_y_exponential_jax(x_example, xi_example, F_example, k_example)
    print(f"Input x: {x_example}")
    print(f"Input x_i array: {xi_example}")
    print(f"Total F: {F_example}")
    print(f"Parameter k: {k_example}")
    print(f"Calculated y_i values: {y_values}")
    print(f"Sum of y_i values: {jnp.sum(y_values)}") # Should be close to F_example

    # --- Demonstrate Automatic Differentiation ---

    # We want to calculate ∂yi/∂x.
    # JAX's jacobian function can compute this directly.
    # jacobian(func)(arg) computes the Jacobian of func with respect to arg.
    # Here, func is calculate_y_exponential_jax, and arg is x_val (the first argument, index 0).

    # To get d(y_i)/dx for all y_i, we compute the Jacobian of the function
    # with respect to its first argument (x_val).
    jacobian_of_y_wrt_x = jax.jacobian(calculate_y_exponential_jax, argnums=0)

    # Calculate the Jacobian at the example point
    # Note: jacobian_of_y_wrt_x expects all arguments that calculate_y_exponential_jax expects.
    dydx_values = jacobian_of_y_wrt_x(x_example, xi_example, F_example, k_example)

    print(f"\n--- Automatic Differentiation Example ---")
    print(f"Jacobian ∂y_i/∂x at x={x_example}:")
    for i, val in enumerate(dydx_values):
        print(f"  ∂y_{i}/∂x = {val:.4f}")

    # You can also get the gradient of a scalar function.
    # For example, if you were interested in the gradient of the sum of y_i with respect to x.
    # (Which should be 0 if sum(y_i) is always F, but let's test for a single y_i)
    def get_first_y(x_val_scalar, x_i_array_scalar, F_val_scalar, k_val_scalar):
        """Helper to get just the first y_i for scalar gradient demo."""
        return calculate_y_exponential_jax(x_val_scalar, x_i_array_scalar, F_val_scalar, k_val_scalar)[0]

    # Gradient of the first y_i component with respect to x
    grad_y0_wrt_x = jax.grad(get_first_y, argnums=0)
    dy0dx_value = grad_y0_wrt_x(x_example, xi_example, F_example, k_example)
    print(f"\nGradient ∂y_0/∂x at x={x_example} (using jax.grad on a scalar output): {dy0dx_value:.4f}")
    assert jnp.isclose(dydx_values[0], dy0dx_value) # Should be the same as the first element of the Jacobian

    # --- Example with a different k to show its effect ---
    k_narrow_example = jnp.array(5.0, dtype=jnp.float32) # Larger k, more focused
    y_values_narrow = calculate_y_exponential_jax(x_example, xi_example, F_example, k_narrow_example)
    print(f"\nExample with k={k_narrow_example}:")
    print(f"Calculated y_i values: {y_values_narrow}")
    print(f"Sum of y_i values: {jnp.sum(y_values_narrow)}")

    dydx_values_narrow = jacobian_of_y_wrt_x(x_example, xi_example, F_example, k_narrow_example)
    print(f"Jacobian ∂y_i/∂x at x={x_example} (k={k_narrow_example}):")
    for i, val in enumerate(dydx_values_narrow):
        print(f"  ∂y_{i}/∂x = {val:.4f}")


def gaussian_load(F, x0, sigma, x, h):
    """Return q_i (units N·m⁻¹) as a JAX array."""
    weights = jnp.exp(-0.5 * ((x - x0)/sigma)**2)
    # exact discrete normalisation
    delta = weights / (h * jnp.sum(weights))
    return F * delta
