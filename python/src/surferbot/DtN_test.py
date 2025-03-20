import jax.numpy as jnp
import unittest
from DtN import DtN_generator

# Define harmonic functions and their y-derivatives
a = 1.0
harmonic_tests = [
    {
        "phi": lambda x, y: jnp.log(jnp.sqrt((x)**2+(y-a)**2)),
        "dphi_dy": lambda x, y: (y-a)/(x**2 + (y-a)**2)
    },
    
]

class TestDtN(unittest.TestCase):
    def test_DtN_accuracy(self):
        Delta_x = 1e-1
        L = 50; N = int(2*L/Delta_x) + 1
        #N = 201  # Number of points
        #Delta_x = 1 / jnp.float32(N-1)  # Discretization step

        x_vals = jnp.linspace(-L, L, N)  # x-grid at y=0
        
        for test in harmonic_tests:
            phi_func = test["phi"]
            dphi_dy_func = test["dphi_dy"]

            phi_values = phi_func(x_vals, 0)  # Sample the function at y=0
            expected_derivatives = dphi_dy_func(x_vals, 0)  # Compute true derivatives at y=0

            DtN = DtN_generator(Delta_x=Delta_x, N=N)  # Generate the DtN matrix
            computed_derivatives = DtN @ phi_values  # Apply the matrix

            # Assert that the computed derivatives match expected values within tolerance
            max_error = jnp.max(jnp.abs(computed_derivatives - expected_derivatives))
            l2_error = jnp.linalg.norm(computed_derivatives-expected_derivatives, ord=2)
            print(f"L2 norm: {l2_error}")
            self.assertTrue(max_error < 1e-2, f"Max error {max_error} exceeded tolerance")

if __name__ == "__main__":
    unittest.main()
