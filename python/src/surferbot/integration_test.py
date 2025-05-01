import math
import unittest
import jax.numpy as jnp

# Import the Simpson weights function
from integration import simpson_weights  # adjust path as needed

# Test function and its exact integral
def f(x):
    return jnp.sin(x)

def exact_integral(a, b):
    return -jnp.cos(b) + jnp.cos(a)

class TestSimpsonWeights(unittest.TestCase):

    def test_integration_weights_accuracy(self):
        """
        For several grid sizes, check that the Simpson rule approximation for f(x) = sin(x)
        over [0, 1] is within a tolerance that scales as h^4.
        """
        a, b = 0.0, 1.0
        for N in [9, 17, 33, 65, 129]:  # Must be odd
            h = (b - a) / (N - 1)
            x = jnp.linspace(a, b, N)
            w = simpson_weights(N, h)
            approx = jnp.dot(w, f(x))
            error = jnp.abs(approx - exact_integral(a, b))
            tolerance = (10 * h) ** 4
            self.assertLess(
                error,
                tolerance,
                msg=f"For N={N}, error {error:.3e} exceeded tolerance {tolerance:.3e}"
            )
    '''
    def test_convergence_rate(self):
        """
        Compute the empirical convergence rate on a sequence of grid sizes and
        assert that the average rate is close to 4.
        """
        a, b = 0.0, 1.0
        Ns = [9, 17, 33, 65, 129]
        errors = []
        hs = []

        for N in Ns:
            h = (b - a) / (N - 1)
            x = jnp.linspace(a, b, N)
            w = simpson_weights(N, h)
            approx = jnp.dot(w, f(x))
            err = jnp.abs(approx - exact_integral(a, b))
            errors.append(err)
            hs.append(h)

        rates = []
        for i in range(1, len(errors)):
            rate = jnp.log(errors[i - 1] / errors[i]) / jnp.log(hs[i - 1] / hs[i])
            rates.append(rate)

        print(rates)

        avg_rate = float(jnp.mean(jnp.array(rates)))
        expected_order = 4.0
        rel_tol = 0.4
        print("Convergence rates:", rates)
        self.assertTrue(
            math.isclose(avg_rate, expected_order, rel_tol=rel_tol),
            msg=f"Empirical convergence rate {avg_rate:.2f} deviates from expected {expected_order}"
        )
    '''

if __name__ == '__main__':
    unittest.main()
