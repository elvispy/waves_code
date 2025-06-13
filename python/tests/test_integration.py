import pytest
import numpy as np
import jax.numpy as jnp
from surferbot.integration import simpson_weights

# Dictionary of test functions and their exact integrals over [0,10]
TEST_FUNCTIONS = {
    "sin(x)": {
        "f": lambda x: jnp.sin(x),
        "integral": lambda a, b: -jnp.cos(b) + jnp.cos(a)
    },
    "x^2": {
        "f": lambda x: x**2,
        "integral": lambda a, b: (b**3 - a**3) / 3.0
    },
    "exp(x)": {
        "f": lambda x: jnp.exp(x),
        "integral": lambda a, b: jnp.exp(b) - jnp.exp(a)
    },
    "1/(1 + x^2)": {
        "f": lambda x: 1 / (1 + x**2),
        "integral": lambda a, b: jnp.arctan(b) - jnp.arctan(a)
    },
    "x^3": {
        "f": lambda x: x**3,
        "integral": lambda a, b: (b**4 - a**4) / 4.0
    },
    "log(1 + x)": {
        "f": lambda x: jnp.log(1 + x),
        "integral": lambda a, b: (1 + b) * jnp.log(1 + b) - b
                                - ((1 + a) * jnp.log(1 + a) - a)
    },
    "sqrt(x)": {
        "f": lambda x: jnp.sqrt(x),
        "integral": lambda a, b: (2 / 3) * (b ** 1.5 - a ** 1.5)
    },
    "sin(x) * cos(x)": {
        "f": lambda x: jnp.sin(x) * jnp.cos(x),
        "integral": lambda a, b: jnp.sin(b) ** 2 / 2 - jnp.sin(a) ** 2 / 2
    },
    "1 / (1 + x)": {
        "f": lambda x: 1 / (1 + x),
        "integral": lambda a, b: jnp.log(1 + b) - jnp.log(1 + a)
    }
}


@pytest.mark.parametrize("name, funcs", TEST_FUNCTIONS.items())
@pytest.mark.parametrize("N", [9, 17, 33, 65])
def test_uniform_float_grid(name, funcs, N):
    """
    Simpson’s rule on a uniform grid (h as float) has error ~ O(h^4).
    """
    a, b = 0.0, 10.0
    f = funcs["f"]
    exact = funcs["integral"](a, b)

    h = (b - a) / (N - 1)
    x = jnp.linspace(a, b, N)
    w = simpson_weights(N, h)
    approx = jnp.dot(w, f(x))
    error = jnp.abs(approx - exact)
    tol = (5 * h) ** 4

    assert error < tol, f"{name}: N={N}, err={error:.3e} > tol={tol:.3e}"


@pytest.mark.parametrize("name, funcs", TEST_FUNCTIONS.items())
@pytest.mark.parametrize("N", [9, 17, 33, 65])
def test_uniform_array_grid(name, funcs, N):
    """
    Simpson’s rule on a uniform grid (h as coordinate array) matches float-h.
    """
    a, b = 0.0, 10.0
    f = funcs["f"]
    exact = funcs["integral"](a, b)

    x = jnp.linspace(a, b, N)
    w = simpson_weights(N, x)        # pass the grid array
    approx = jnp.dot(w, f(x))

    h = (b - a) / (N - 1)
    tol = (5 * h) ** 4
    error = jnp.abs(approx - exact)

    assert error < tol, f"{name}(array): N={N}, err={error:.3e} > tol={tol:.3e}"


def test_nonuniform_weights_sum_to_length_approx():
    """
    On a non-uniform grid, Simpson weights sum to approximately the interval length,
    within the scale of the maximum node jitter.
    """
    a, b = 0.0, 1.0
    N = 55  # odd number of points
    rng = np.random.RandomState(0)

    # start with a uniform grid then add jitter
    x_uniform = np.linspace(a, b, N)
    jitter = rng.uniform(-0.02, 0.02, size=N)
    x_jit = np.sort(x_uniform + jitter)
    x = jnp.array(x_jit)

    # compute weights and their sum
    w = simpson_weights(N, x)
    total = float(jnp.sum(w))

    # allow error up to the maximum jitter magnitude
    tol = float(np.max(np.abs(jitter)))

    assert abs(total - (b - a)) < tol, (
        f"Total weight {total:.4f} differs from {b - a:.4f} "
        f"by more than max_jitter={tol:.4f}"
    )


def test_nonuniform_sin_accuracy():
    """
    On a mildly non-uniform grid over [0,10], sin(x) still integrates with near–Simpson accuracy.
    """
    a, b = 0.0, 10.0
    N = 65  # odd
    rng = np.random.RandomState(1)

    x_uniform = np.linspace(a, b, N)
    jitter = rng.uniform(-0.02, 0.02, size=N)
    x_jit = np.sort(x_uniform + jitter)
    x = jnp.array(x_jit)

    w = simpson_weights(N, x)
    approx = jnp.dot(w, jnp.sin(x))
    exact = -jnp.cos(b) + jnp.cos(a)

    h = (b - a) / (N - 1)
    tol = (5 * h) ** 4
    error = jnp.abs(approx - exact)

    assert error < tol, f"sin(x) nonuniform: err={error:.3e} > tol={tol:.3e}"
