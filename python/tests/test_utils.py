import unittest
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import numpy as np
from surferbot.utils import solve_tensor_system, solve_k, gaussian_load

class TestSolveTensorSystem(unittest.TestCase):
    """Test suite for the solve_tensor_system function."""

    def setUp(self):
        """Create a JAX random key for use in all tests."""
        self.key = jax.random.PRNGKey(42)

    # --- Original 5 Tests ---
    
    def test_square_dense_system_real(self):
        """1. Tests recovery of a known solution for a diagonally dominant dense system."""
        key_A, key_x = jax.random.split(self.key)
        dim = 5
        A = jax.random.normal(key_A, (dim, dim)) + jnp.eye(dim) * 10.0
        x_expected = jax.random.normal(key_x, (dim,))
        b = A @ x_expected
        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-5)

    def test_rectangular_dense_system_real(self):
        """2. Tests recovery of a known solution for a well-conditioned rectangular system."""
        key_A, key_x = jax.random.split(self.key)
        m, n = 8, 5
        A = jax.random.normal(key_A, (m, n)) + jnp.eye(m, n) * 10.0
        x_expected = jax.random.normal(key_x, (n,))
        b = A @ x_expected
        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-5)

    def test_square_sparse_system_real(self):
        """3. Tests recovery of a known solution for a diagonally dominant sparse system."""
        key_A, key_x = jax.random.split(self.key)
        dim = 10
        A_dense = jax.random.normal(key_A, (dim, dim)) * 0.1 + jnp.eye(dim) * 10.0
        A_sparse = jsparse.BCOO.fromdense(A_dense)
        x_expected = jax.random.normal(key_x, (dim,))
        b = A_dense @ x_expected
        x_actual = solve_tensor_system(A_sparse, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-5)

    def test_shape_mismatch_error(self):
        """4. Verifies ValueError for incompatible A and b shapes."""
        A = jnp.zeros((3, 3, 2))
        b = jnp.zeros((4,))
        # This is the corrected, more robust pattern
        with self.assertRaises(ValueError) as cm:
            solve_tensor_system(A, b)
        
        self.assertEqual(str(cm.exception), "b.shape must equal A.shape[:b.ndim]")

    def test_gmres_convergence_error(self):
        """5. Checks RuntimeError when GMRES fails to converge."""
        A_singular_sparse = jsparse.BCOO.fromdense(jnp.ones((2, 2)))
        b = jnp.array([1.0, 2.0])
        with self.assertRaisesRegex(RuntimeError, "GMRES failed to converge"):
            solve_tensor_system(A_singular_sparse, b, maxiter=1)
            
    # --- 5 New Multidimensional Tests ---

    def test_md_dense_square_flat(self):
        """6. MD Test: Dense A(B,C,D), b(B,C) -> x(D). Flattened system is square."""
        key_A, key_x = jax.random.split(self.key)
        shape_b, shape_x = (2, 3), (6,)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x)
        
        A = jax.random.normal(key_A, (x_size, *shape_x)) + jnp.eye(x_size).reshape(x_size, *shape_x) * 10.0
        A = A.reshape((*shape_b, *shape_x))
        x_expected = jax.random.normal(key_x, shape_x)
        
        # Calculate b = A_flat @ x_flat
        b = (A.reshape(b_size, x_size) @ x_expected.flatten()).reshape(shape_b)
        
        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-5)

    def test_md_dense_rectangular_flat(self):
        """7. MD Test: Dense A(B,C,D,E), b(B,C) -> x(D,E). Flattened system is rectangular."""
        key_A, key_x = jax.random.split(self.key)
        shape_b, shape_x = (4, 5), (2, 3) # Flat: A(20, 6), x(6), b(20)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x)
        
        A_flat_random = jax.random.normal(key_A, (b_size, x_size))
        x_expected = jax.random.normal(key_x, shape_x)
        A = A_flat_random.reshape(*shape_b, *shape_x)

        b = (A_flat_random @ x_expected.flatten()).reshape(shape_b)

        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-5)

    def test_md_sparse_square_flat(self):
        """8. MD Test: Sparse A(B,C,D), b(B) -> x(C,D). Flattened system is square."""
        key_A, key_x = jax.random.split(self.key)
        shape_b, shape_x = (4,), (5, 5) # Flat: A(4, 25), x(25) -> error. A(4, 5, 5), b(4) -> x(5,5). Flat: A(4, 25)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x)
        
        # Let's make the flattened system square for GMRES
        shape_b, shape_x = (5,), (5,) # A(5,5), b(5) -> x(5). Flat: A(5,5).
        A_shape = (*shape_b, *shape_x)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x) # 5, 5
        
        A_dense = jax.random.normal(key_A, A_shape) + jnp.eye(x_size).reshape(A_shape) * 10.0
        A_sparse = jsparse.BCOO.fromdense(A_dense)
        x_expected = jax.random.normal(key_x, shape_x)

        b = (A_dense.reshape(b_size, x_size) @ x_expected.flatten()).reshape(shape_b)

        x_actual = solve_tensor_system(A_sparse, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-4)

    def test_md_dense_scalar_b(self):
        """9. MD Test: Dense A(D,E), b() -> x(D,E). b is a scalar."""
        key_A, key_x = jax.random.split(self.key)
        shape_x = (4, 5)
        x_size = np.prod(shape_x)

        A = jax.random.normal(key_A, shape_x)
        # For a scalar b, the flattened system is 1-row. We seek min-norm solution.
        # x_expected = A_flat.T @ inv(A_flat @ A_flat.T) @ b
        x_flat_expected = jnp.linalg.pinv(A.reshape(1, x_size)) @ jnp.array([5.0])
        x_expected = x_flat_expected.reshape(shape_x)
        b = jnp.array(5.0) # scalar b

        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-5)

    def test_md_dense_overdetermined_flat(self):
        """10. MD Test: Dense A(B,C,D), b(B,C) -> x(D). Flattened system is overdetermined."""
        key_A, key_x = jax.random.split(self.key)
        shape_b, shape_x = (8, 2), (3,) # Flat: A(16, 3), x(3), b(16)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x)

        A_flat_random = jax.random.normal(key_A, (b_size, x_size))
        A = A_flat_random.reshape(*shape_b, *shape_x)
        x_expected = jax.random.normal(key_x, shape_x)
        
        b = (A_flat_random @ x_expected.flatten()).reshape(shape_b)

        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-5)

    # --- 5 New Complex64 Tests ---

    def _get_complex_random(self, key, shape):
        key_r, key_i = jax.random.split(key)
        return (jax.random.normal(key_r, shape, dtype=jnp.float32) + 
                1j * jax.random.normal(key_i, shape, dtype=jnp.float32)).astype(jnp.complex64)

    def test_square_dense_system_complex(self):
        """11. Complex Test: Square dense system with complex64 dtype."""
        key_A, key_x = jax.random.split(self.key)
        dim = 5
        A = self._get_complex_random(key_A, (dim, dim)) + jnp.eye(dim) * 10.0
        x_expected = self._get_complex_random(key_x, (dim,))
        b = A @ x_expected
        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-5)

    def test_rectangular_dense_system_complex(self):
        """12. Complex Test: Rectangular dense system with complex64 dtype."""
        key_A, key_x = jax.random.split(self.key)
        m, n = 8, 5
        A = self._get_complex_random(key_A, (m, n)) + jnp.eye(m, n) * 10.0
        x_expected = self._get_complex_random(key_x, (n,))
        b = A @ x_expected
        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-5)

    def test_square_sparse_system_complex(self):
        """13. Complex Test: Square sparse system with complex64 dtype."""
        key_A, key_x = jax.random.split(self.key)
        dim = 10
        A_dense = self._get_complex_random(key_A, (dim, dim)) * 0.1 + jnp.eye(dim) * 10.0
        A_sparse = jsparse.BCOO.fromdense(A_dense)
        x_expected = self._get_complex_random(key_x, (dim,))
        b = A_dense @ x_expected
        x_actual = solve_tensor_system(A_sparse, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-4)
    
    def test_md_dense_system_complex(self):
        """14. Complex Test: Multidimensional dense system with complex64 dtype."""
        key_A, key_x = jax.random.split(self.key)
        shape_b, shape_x = (4, 5), (2, 3)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x)

        A = self._get_complex_random(key_A, (*shape_b, *shape_x))
        x_expected = self._get_complex_random(key_x, shape_x)

        b = (A.reshape(b_size, x_size) @ x_expected.flatten()).reshape(shape_b)

        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-5)

    def test_md_sparse_system_complex(self):
        """15. Complex Test: MD sparse system A(4,5,4,5) with complex64 dtype."""
        key_A, key_x = jax.random.split(self.key)

        # 1. Define shapes per the request. This creates a square flattened system.
        shape_b, shape_x = (4, 5), (4, 5)
        # The flattened system will have m = 20, n = 20.
        flat_dim = np.prod(shape_b)

        # 2. Create a diagonally dominant matrix in its flattened (20, 20) form
        A_flat_random = self._get_complex_random(key_A, (flat_dim, flat_dim)) * 0.1
        identity_flat = jnp.eye(flat_dim, dtype=jnp.complex64) * 10.0
        A_flat = A_flat_random + identity_flat

        # 3. Reshape to the final MD shape and create the sparse version
        A_dense = A_flat.reshape(*shape_b, *shape_x)
        A_sparse = jsparse.BCOO.fromdense(A_dense)

        # 4. Create the known complex solution x and corresponding b
        x_expected = self._get_complex_random(key_x, shape_x)
        # Calculate b = A_flat @ x_flat, then reshape b
        b = (A_flat @ x_expected.flatten()).reshape(shape_b)

        # 5. Solve the system using the sparse MD matrix and verify the result
        x_actual = solve_tensor_system(A_sparse, b)
        np.testing.assert_allclose(x_expected, x_actual, atol=1e-4)
        

class TestGaussianLoad(unittest.TestCase):
    """Tests for the `gaussian_load` discrete Gaussian-point-load function."""

    def setUp(self):
        # A simple non-uniform grid
        self.x = jnp.array([0.0, 0.5, 1.5, 3.0])
        self.x0 = 1.2
        self.sigma = 0.4

    def _weights(self, x):
        """Recompute the trapezoidal weights used inside gaussian_load."""
        dx = jnp.diff(x)
        w_mid = 0.5 * (dx[:-1] + dx[1:])
        return jnp.concatenate([0.5 * dx[:1], w_mid, 0.5 * dx[-1:]])

    def test_sum_to_one(self):
        """The discrete integral of the load density should equal 1."""
        w = self._weights(self.x)
        q = gaussian_load(self.x0, self.sigma, self.x)
        total = jnp.sum(q * w)
        # Sum over nodes times their weights ≈ 1
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_integral_gradient_zero(self):
        """Since the integral is constant, its derivative w.r.t x0 and sigma is zero."""
        w = self._weights(self.x)

        # Function that maps x0 → discrete integral
        integral_wrt_x0 = lambda x0: jnp.sum(gaussian_load(x0, self.sigma, self.x) * w)
        grad_x0 = jax.grad(integral_wrt_x0)(self.x0)
        self.assertAlmostEqual(grad_x0, 0.0, places=6)

        # Function that maps sigma → discrete integral
        integral_wrt_sigma = lambda sigma: jnp.sum(gaussian_load(self.x0, sigma, self.x) * w)
        grad_sigma = jax.grad(integral_wrt_sigma)(self.sigma)
        self.assertAlmostEqual(grad_sigma, 0.0, places=6)

    def test_jacobian_no_nans(self):
        """The Jacobian of q w.r.t. x0 and sigma should exist and contain no NaNs."""
        # dq/dx0
        jac_x0 = jax.jacobian(lambda x0: gaussian_load(x0, self.sigma, self.x))(self.x0)
        self.assertFalse(jnp.any(jnp.isnan(jac_x0)))

        # dq/dsigma
        jac_sigma = jax.jacobian(lambda sigma: gaussian_load(self.x0, sigma, self.x))(self.sigma)
        self.assertFalse(jnp.any(jnp.isnan(jac_sigma)))



class TestSolveK(unittest.TestCase):
    """Tests for the `solve_k` dispersion‐root finder and its AD."""

    def setUp(self):
        # fixed PRNG for reproducibility
        self.key = jax.random.PRNGKey(1234)

    def _dispersion_eq(self, k, omega, g, H, nu, sigma, rho):
        tanh_kH = jnp.tanh(k * H)
        lhs = k * tanh_kH * g
        rhs = (
            (-sigma / rho) * k**3 * tanh_kH
            + omega**2
            - 4j * nu * omega * k**2
        )
        return lhs - rhs

    def test_dispersion_relation(self):
        """solve_k should return k satisfying the dispersion relation (residual ≈ 0)."""
        # split RNG for each parameter
        keys = jax.random.split(self.key, 6)
        omega = jax.random.uniform(keys[0], (), minval=0.1,  maxval=10.0)
        g     = jax.random.uniform(keys[1], (), minval=1.0,  maxval=20.0)
        H     = jax.random.uniform(keys[2], (), minval=0.1,  maxval=5.0)
        nu    = jax.random.uniform(keys[3], (), minval=1e-8, maxval=1e-3)
        sigma = jax.random.uniform(keys[4], (), minval=1e-3, maxval=1.0)
        rho   = jax.random.uniform(keys[5], (), minval=100.0, maxval=2000.0)

        # solve for k
        k = solve_k(omega, g, H, nu, sigma, rho)

        # compute residual
        res = self._dispersion_eq(k, omega, g, H, nu, sigma, rho)

        # verify both real and imag parts ≈ 0
        np.testing.assert_allclose(jnp.real(res), 0.0, atol=1e-6)
        np.testing.assert_allclose(jnp.imag(res), 0.0, atol=1e-6)

    def test_gradient_not_nan(self):
        """The ω-derivatives of k (real and imag parts) must not be NaN."""
        keys = jax.random.split(self.key, 6)
        omega = jax.random.uniform(keys[0], (), minval=0.1,  maxval=10.0)
        g     = jax.random.uniform(keys[1], (), minval=1.0,  maxval=20.0)
        H     = jax.random.uniform(keys[2], (), minval=0.1,  maxval=5.0)
        nu    = jax.random.uniform(keys[3], (), minval=1e-8, maxval=1e-3)
        sigma = jax.random.uniform(keys[4], (), minval=1e-3, maxval=1.0)
        rho   = jax.random.uniform(keys[5], (), minval=100.0, maxval=2000.0)

        # wrap solve_k as a function of omega alone
        k_fn = lambda w: solve_k(w, g, H, nu, sigma, rho)

        # compute real/imag gradients
        dk_real = jax.grad(lambda w: jnp.real(k_fn(w)))(omega)
        dk_imag = jax.grad(lambda w: jnp.imag(k_fn(w)))(omega)

        # assert neither is NaN
        self.assertFalse(jnp.isnan(dk_real).item(), "d(Re k)/dω is NaN")
        self.assertFalse(jnp.isnan(dk_imag).item(), "d(Im k)/dω is NaN")
if __name__ == '__main__':
    unittest.main()