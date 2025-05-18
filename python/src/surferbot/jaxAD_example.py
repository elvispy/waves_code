import jax
import jax.numpy as jnp

def build_matrix(params):
    a, b, c = params
    return jnp.array([
        [a,   1.0 + 0j, 2.0 + 0j],
        [0.0 + 0j, b,   1.0 + 0j],
        [1.0 + 0j, 1.0 + 0j, c],
        [2.0 + 0j, 0.0 + 0j, 1.0 + 0j],
    ], dtype=jnp.complex64)

def build_vector(params):
    a, b, c = params
    return jnp.array([1.0 + a, 2.0 + b, 3.0 + c, 4.0], dtype=jnp.complex64)

def objective(params):
    A = build_matrix(params)
    b = build_vector(params)
    x = jnp.linalg.lstsq(A, b, rcond=None)[0]
    return jnp.real(jnp.sum(x))  # real scalar output for grad

# Must split real and imaginary parts if you want complex params
# Here we treat real-valued parameters that produce complex A and b
grad_fn = jax.grad(objective)

params = jnp.array([10.0, 20.0, 30.0])
print("Loss (sum of x):", objective(params))
print("Gradient:", grad_fn(params))
