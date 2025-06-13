import pytest
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import numpy as np

from surferbot.utils import solve_tensor_system, solve_k, gaussian_load

@pytest.fixture
def prng_key():
    """Fixed JAX PRNG key for reproducibility."""
    return jax.random.PRNGKey(42)

@pytest.fixture
def simple_grid():
    """Non-uniform 1D grid plus its trapezoidal weights."""
    x = jnp.array([0.0, 0.5, 1.5, 3.0])
    dx = jnp.diff(x)
    w_mid = 0.5 * (dx[:-1] + dx[1:])
    w = jnp.concatenate([0.5 * dx[:1], w_mid, 0.5 * dx[-1:]])
    return x, w

@pytest.fixture
def solve_k_params(prng_key):
    """Random positive params for solve_k(ω, g, H, ν, σ, ρ)."""
    keys = jax.random.split(prng_key, 6)
    omega = jax.random.uniform(keys[0], (), minval=0.1,  maxval=10.0)
    g     = jax.random.uniform(keys[1], (), minval=1.0,  maxval=20.0)
    H     = jax.random.uniform(keys[2], (), minval=0.1,  maxval=5.0)
    nu    = jax.random.uniform(keys[3], (), minval=1e-8, maxval=1e-3)
    sigma = jax.random.uniform(keys[4], (), minval=1e-3, maxval=1.0)
    rho   = jax.random.uniform(keys[5], (), minval=100.0, maxval=2000.0)
    return omega, g, H, nu, sigma, rho


class TestSolveTensorSystem:
    def test_square_dense_system_real(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        dim = 5
        A = jax.random.normal(key_A, (dim, dim)) + jnp.eye(dim) * 10.0
        x_expected = jax.random.normal(key_x, (dim,))
        b = A @ x_expected
        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-5)

    def test_rectangular_dense_system_real(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        m, n = 8, 5
        A = jax.random.normal(key_A, (m, n)) + jnp.eye(m, n) * 10.0
        x_expected = jax.random.normal(key_x, (n,))
        b = A @ x_expected
        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-5)

    def test_square_sparse_system_real(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        dim = 10
        A_dense = (jax.random.normal(key_A, (dim, dim)) * 0.1
                   + jnp.eye(dim) * 10.0)
        A_sparse = jsparse.BCOO.fromdense(A_dense)
        x_expected = jax.random.normal(key_x, (dim,))
        b = A_dense @ x_expected
        x_actual = solve_tensor_system(A_sparse, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-5)

    def test_shape_mismatch_error(self):
        A = jnp.zeros((3, 3, 2))
        b = jnp.zeros((4,))
        with pytest.raises(ValueError) as exc:
            solve_tensor_system(A, b)
        assert str(exc.value) == "b.shape must equal A.shape[:b.ndim]"

    def test_gmres_convergence_error(self):
        A_sing = jsparse.BCOO.fromdense(jnp.ones((2, 2)))
        b = jnp.array([1.0, 2.0])
        with pytest.raises(RuntimeError, match="GMRES failed to converge"):
            solve_tensor_system(A_sing, b, maxiter=1)

    def test_md_dense_square_flat(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        shape_b, shape_x = (2, 3), (6,)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x)

        A = (jax.random.normal(key_A, (x_size, *shape_x))
             + jnp.eye(x_size).reshape(x_size, *shape_x) * 10.0)
        A = A.reshape((*shape_b, *shape_x))
        x_expected = jax.random.normal(key_x, shape_x)
        b = (A.reshape(b_size, x_size) @ x_expected.flatten()).reshape(shape_b)

        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-5)

    def test_md_dense_rectangular_flat(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        shape_b, shape_x = (4, 5), (2, 3)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x)

        A_flat = jax.random.normal(key_A, (b_size, x_size))
        x_expected = jax.random.normal(key_x, shape_x)
        A = A_flat.reshape(*shape_b, *shape_x)
        b = (A_flat @ x_expected.flatten()).reshape(shape_b)

        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-5)

    def test_md_sparse_square_flat(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        shape_b, shape_x = (5,), (5,)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x)

        A_dense = (jax.random.normal(key_A, (*shape_b, *shape_x))
                   + jnp.eye(x_size).reshape(*shape_b, *shape_x) * 10.0)
        A_sparse = jsparse.BCOO.fromdense(A_dense)
        x_expected = jax.random.normal(key_x, shape_x)
        b = (A_dense.reshape(b_size, x_size) @ x_expected.flatten()).reshape(shape_b)

        x_actual = solve_tensor_system(A_sparse, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-4)

    def test_md_dense_scalar_b(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        shape_x = (4, 5)
        x_size = np.prod(shape_x)

        A = jax.random.normal(key_A, shape_x)
        x_flat = (jnp.linalg.pinv(A.reshape(1, x_size))
                  @ jnp.array([5.0]))
        x_expected = x_flat.reshape(shape_x)
        b = jnp.array(5.0)  # scalar b as a 0-d array

        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-5)

    def test_md_dense_overdetermined_flat(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        shape_b, shape_x = (8, 2), (3,)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x)

        A_flat = jax.random.normal(key_A, (b_size, x_size))
        x_expected = jax.random.normal(key_x, shape_x)
        A = A_flat.reshape(*shape_b, *shape_x)
        b = (A_flat @ x_expected.flatten()).reshape(shape_b)

        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-5)

    def _get_complex_random(self, key, shape):
        key_r, key_i = jax.random.split(key)
        return (
            jax.random.normal(key_r, shape, dtype=jnp.float32)
            + 1j * jax.random.normal(key_i, shape, dtype=jnp.float32)
        ).astype(jnp.complex64)

    def test_square_dense_system_complex(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        dim = 5
        A = self._get_complex_random(key_A, (dim, dim)) + jnp.eye(dim) * 10.0
        x_expected = self._get_complex_random(key_x, (dim,))
        b = A @ x_expected
        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-5)

    def test_rectangular_dense_system_complex(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        m, n = 8, 5
        A = self._get_complex_random(key_A, (m, n)) + jnp.eye(m, n) * 10.0
        x_expected = self._get_complex_random(key_x, (n,))
        b = A @ x_expected
        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-5)

    def test_square_sparse_system_complex(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        dim = 10
        A_dense = (
            self._get_complex_random(key_A, (dim, dim)) * 0.1
            + jnp.eye(dim) * 10.0
        )
        A_sparse = jsparse.BCOO.fromdense(A_dense)
        x_expected = self._get_complex_random(key_x, (dim,))
        b = A_dense @ x_expected
        x_actual = solve_tensor_system(A_sparse, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-4)

    def test_md_dense_system_complex(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        shape_b, shape_x = (4, 5), (2, 3)
        b_size, x_size = np.prod(shape_b), np.prod(shape_x)

        A = self._get_complex_random(key_A, (*shape_b, *shape_x))
        x_expected = self._get_complex_random(key_x, shape_x)
        b = (A.reshape(b_size, x_size) @ x_expected.flatten()).reshape(shape_b)

        x_actual = solve_tensor_system(A, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-5)

    def test_md_sparse_system_complex(self, prng_key):
        key_A, key_x = jax.random.split(prng_key)
        shape_b, shape_x = (4, 5), (4, 5)
        flat_dim = np.prod(shape_b)

        A_flat = (
            self._get_complex_random(key_A, (flat_dim, flat_dim)) * 0.1
            + jnp.eye(flat_dim, dtype=jnp.complex64) * 10.0
        )
        A_dense = A_flat.reshape(*shape_b, *shape_x)
        A_sparse = jsparse.BCOO.fromdense(A_dense)
        x_expected = self._get_complex_random(key_x, shape_x)
        b = (A_flat @ x_expected.flatten()).reshape(shape_b)

        x_actual = solve_tensor_system(A_sparse, b)
        np.testing.assert_allclose(x_actual, x_expected, atol=1e-4)


class TestGaussianLoad:
    def test_sum_to_one(self, simple_grid):
        x, w = simple_grid
        x0, sigma = 1.2, 0.4
        q = gaussian_load(x0, sigma, x)
        total = jnp.sum(q * w)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_integral_gradient_zero(self, simple_grid):
        x, w = simple_grid
        x0, sigma = 1.2, 0.4

        f_x0 = lambda x0: jnp.sum(gaussian_load(x0, sigma, x) * w)
        grad_x0 = jax.grad(f_x0)(x0)
        assert grad_x0 == pytest.approx(0.0, abs=1e-6)

        f_sigma = lambda sigma: jnp.sum(gaussian_load(x0, sigma, x) * w)
        grad_sigma = jax.grad(f_sigma)(sigma)
        assert grad_sigma == pytest.approx(0.0, abs=1e-6)

    def test_jacobian_no_nans(self, simple_grid):
        x, w = simple_grid
        x0, sigma = 1.2, 0.4

        jac_x0 = jax.jacobian(lambda x0: gaussian_load(x0, sigma, x))(x0)
        assert not jnp.any(jnp.isnan(jac_x0))

        jac_sigma = jax.jacobian(lambda sigma: gaussian_load(x0, sigma, x))(sigma)
        assert not jnp.any(jnp.isnan(jac_sigma))


class TestSolveK:
    def test_dispersion_relation(self, solve_k_params):
        omega, g, H, nu, sigma, rho = solve_k_params

        def dispersion_eq(k):
            t = jnp.tanh(k * H)
            lhs = k * t * g
            rhs = (-sigma / rho) * k**3 * t + omega**2 - 4j * nu * omega * k**2
            return lhs - rhs

        k = solve_k(omega, g, H, nu, sigma, rho)
        res = dispersion_eq(k)
        np.testing.assert_allclose(res.real, 0.0, atol=1e-6)
        np.testing.assert_allclose(res.imag, 0.0, atol=1e-6)

    def test_gradient_not_nan(self, solve_k_params):
        omega, g, H, nu, sigma, rho = solve_k_params
        k_fn = lambda w: solve_k(w, g, H, nu, sigma, rho)

        dk_real = jax.grad(lambda w: jnp.real(k_fn(w)))(omega)
        dk_imag = jax.grad(lambda w: jnp.imag(k_fn(w)))(omega)

        assert not jnp.isnan(dk_real)
        assert not jnp.isnan(dk_imag)
