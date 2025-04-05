import jax.numpy as jnp
import unittest
from DtN import DtN_generator
import scipy.integrate as spi


# Define harmonic functions and their y-derivatives
a = .01
harmonic_tests = [
    {
        "phi": lambda x: jnp.sin(x) * jnp.exp(-(a*x)**2),
        "phi": lambda x: jnp.cos(x) * jnp.exp(-(a*x)**2),
        "phi": lambda x: jnp.sin(x)
    },   
]

def dtn(x0, phi, Dx):
    integrand = lambda x: (phi(x0) - phi(x)) / ((x - x0) ** 2)

    eps = 2*Dx
    f_evals = jnp.array([phi(x) for x in [x0-2*Dx, x0-Dx, x0, x0+Dx, x0+2*Dx]])
    int_near_x0 = jnp.dot(f_evals, jnp.array([-1.0, -32.0, 66.0, -32.0, -1.0]))/(18*Dx)

    # Use SciPy's to integrate. 
    LIMIT = 1000
    return 1/jnp.pi * (spi.quad(integrand, -jnp.inf, x0-eps, limit=LIMIT)[0] + spi.quad(integrand, x0+eps, jnp.inf, limit=LIMIT)[0] + int_near_x0)


class TestDtN(unittest.TestCase):
    def test_DtN_accuracy(self):
        L = 50; N = 500
        #N = 201  # Number of points
        #Delta_x = 1 / jnp.float32(N-1)  # Discretization step

        x_vals = jnp.linspace(-L, L, N)  # x-grid at y=0
        Delta_x = x_vals[1] - x_vals[0]
        
        for test in harmonic_tests:
            phi_func = test["phi"]
            
            phi_values = phi_func(x_vals)  # Sample the function at y=0
            idx = jnp.linspace(L*0.1, L*0.9, 10, dtype=jnp.int64)
            expected_integrals = jnp.array([dtn(x0, phi_func, Dx=Delta_x) for x0 in x_vals[idx]])

            DtN = DtN_generator(N=N)  # Generate the DtN matrix
            computed_integrals = (DtN/Delta_x @ phi_values)[idx] # Apply the matrix

            # Assert that the computed derivatives match expected values within tolerance
            l2_error = jnp.linalg.norm(computed_integrals-expected_integrals, ord=2)
            #print(f"L2 norm: {l2_error}")
            #print(f"Computed: {computed_integrals}")
            #print(f"Expected: {expected_integrals}")
            self.assertTrue(l2_error < 1e-1, f"L2 norm {l2_error} exceeded tolerance")

if __name__ == "__main__":
    unittest.main()
