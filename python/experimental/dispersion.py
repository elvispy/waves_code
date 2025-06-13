import jax
import jax.numpy as jnp

# Constants
H = 1.0
g = 9.81
sigma = 0.072
rho = 1000.0
nu = 1e-6
omega = 2.0
def dispersion_eq(k):
    tanh_kH = jnp.tanh(k * H)
    lhs = k * tanh_kH * g
    rhs = (-sigma / rho) * k**3 * tanh_kH + omega**2 - 4j * nu * omega * k**2
    return lhs - rhs

# Newton iteration for complex k using fixed number of steps (JAX differentiable)
def solve_k(omega, g, H, nu, sigma, rho, k0=1.0 + 0.0j, num_steps=100):
    def dispersion_eq(k):
        tanh_kH = jnp.tanh(k * H)
        lhs = k * tanh_kH * g
        rhs = (-sigma / rho) * k**3 * tanh_kH + omega**2 - 4j * nu * omega * k**2
        return lhs - rhs

    # Define the gradient functions
    real_grad_fn = jax.grad(lambda k: jnp.real(dispersion_eq(k)))
    imag_grad_fn = jax.grad(lambda k: jnp.imag(dispersion_eq(k)))

    def newton_step(k, _):
        f = dispersion_eq(k)
        df_dk = real_grad_fn(k) + 1j * imag_grad_fn(k)  # evaluate both grads at k
        k_next = k - f / df_dk
        return k_next, None

    k_final, _ = jax.lax.scan(newton_step, k0, None, length=num_steps)
    return k_final




# Define a function of omega that returns real or imaginary part of k
k_fn = lambda omega: solve_k(omega, g, H, nu, sigma, rho)

omega_val = omega
k_val = k_fn(omega_val)
dk_domega_real = jax.grad(lambda w: jnp.real(k_fn(w)))(omega_val)
dk_domega_imag = jax.grad(lambda w: jnp.imag(k_fn(w)))(omega_val)

print("k:", k_val)
print(dispersion_eq(k_val))
print("dk/domega (real):", dk_domega_real)
print("dk/domega (imag):", dk_domega_imag)
