import unittest
import jax.numpy as jnp

# Import your Simpson weights function
from integration import simpson_weights  # adjust path if needed

# Dictionary of test functions and their exact integrals over [0, 1]
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
        "integral": lambda a, b: (1 + b) * jnp.log(1 + b) - b - ((1 + a) * jnp.log(1 + a) - a)
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

class TestSimpsonWeights(unittest.TestCase):

    def test_integration_accuracy_multiple_functions(self):
        """
        For several grid sizes and test functions, ensure Simpson's rule gives results
        accurate to ~h^4 over [0, 1].
        """
        a, b = 0.0, 10.0
        for name, funcs in TEST_FUNCTIONS.items():
            f = funcs["f"]
            exact = funcs["integral"](a, b)
            for N in [9, 17, 33, 65]:
                h = (b - a) / (N - 1)
                x = jnp.linspace(a, b, N)
                w = simpson_weights(N, h)
                approx = jnp.dot(w, f(x))
                error = jnp.abs(approx - exact)
                tolerance = (5*h)**4
                self.assertLess(
                    error,
                    tolerance,
                    msg=f"{name}: For N={N}, error {error:.3e} > tolerance {tolerance:.3e}"
                )

if __name__ == '__main__':
    unittest.main()
