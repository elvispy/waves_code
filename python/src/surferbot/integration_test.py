import math
import unittest
import jax.numpy as jnp

# Import the integration_weights function from your module (adjust the path as needed)
from integration import integration_weights

# Define the test function and its exact integral.
def f(x):
    return jnp.sin(x)

def exact_integral(a, b):
    return -jnp.cos(b) + jnp.cos(a)

class TestIntegrationWeights(unittest.TestCase):

    def test_integration_weights_accuracy(self):
        """
        For several grid sizes check that the quadrature approximation for f(x)=exp(x)
        over [0,1] is within a tolerance that scales as h^4.
        """
        a, b = 0.0, 1.0
        for N in [8, 16, 32, 64, 128]:
            x, w = integration_weights(a, b, N)
            approx = jnp.dot(w, f(x))
            error = jnp.abs(approx - exact_integral(a, b))
            # Compute spacing h and tolerance ~ h^4.
            h = (b - a) / (N - 1)
            tolerance =   (10*h )** 4
            self.assertLess(
                error,
                tolerance,
                msg=f"For N={N}, error {error:.3e} exceeded tolerance {tolerance:.3e}"
            )

    def test_convergence_rate(self):
        """
        Compute the empirical convergence rate on a sequence of grid sizes and
        assert that the average rate is close to 4.
        """
        a, b = 0.0, 1.0
        Ns = [8, 16, 32, 64, 128]
        errors = []
        hs = []

        for N in Ns:
            x, w = integration_weights(a, b, N)
            h = (b - a) / (N - 1)
            hs.append(h)
            approx = jnp.dot(w, f(x))
            err = jnp.abs(approx - exact_integral(a, b))
            errors.append(err)

        # Compute convergence rates between successive grid refinements.
        rates = []
        for i in range(1, len(errors)):
            rate = jnp.log(errors[i-1] / errors[i]) / jnp.log(hs[i-1] / hs[i])
            rates.append(rate)
        avg_rate = float(jnp.mean(jnp.array(rates)))
        expected_order = 4.0
        rel_tol = 0.15  # allow ~15% relative tolerance
        print(rates)
        self.assertTrue(
            math.isclose(avg_rate, expected_order, rel_tol=rel_tol),
            msg=f"Empirical convergence rate {avg_rate:.2f} deviates from expected {expected_order}"
        )

if __name__ == '__main__':
    unittest.main()
