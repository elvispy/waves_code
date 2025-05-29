import unittest
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import numpy as np
from surferbot.utils import solve_tensor_system

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
        
if __name__ == '__main__':
    unittest.main()