import unittest
import jax.numpy as jnp
import scipy.integrate as spi

from surferbot.DtN import DtN_generator

# the same dtn() helper you wrote
def dtn(x0, phi, Dx):
    integrand = lambda x: (phi(x0) - phi(x)) / ((x - x0) ** 2)
    eps = 2*Dx
    # 5-point stencil for near-field
    f_evals = jnp.array([phi(x) for x in [x0-2*Dx, x0-Dx, x0, x0+Dx, x0+2*Dx]])
    int_near_x0 = jnp.dot(
        f_evals,
        jnp.array([-1.0, -32.0, 66.0, -32.0, -1.0])
    ) / (18*Dx)

    LIMIT = 1000
    left  = spi.quad(integrand, -jnp.inf, x0-eps, limit=LIMIT)[0]
    right = spi.quad(integrand, x0+eps, jnp.inf,  limit=LIMIT)[0]
    return 1/jnp.pi * (left + right + int_near_x0)


class TestDtN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # constants shared by all tests
        cls.a      = 0.01
        cls.L      = 50
        cls.k      = 200
        cls.nu     = 1e-6
        cls.N      = 500
        cls.x_vals = jnp.linspace(-cls.L, cls.L, cls.N)
        cls.Dx     = cls.x_vals[1] - cls.x_vals[0]
        # pick 10 interior indices
        cls.idx    = jnp.linspace(
            int(0.1*cls.N), int(0.9*cls.N), 10, dtype=jnp.int32
        )
        # build the DtN matrix once
        cls.DtN_mat = DtN_generator(N=cls.N)

    def run_relative_test(self, phi_func):
        # compute “exact” via your dtn()
        expected = jnp.array([
            dtn(self.x_vals[i], phi_func, Dx=self.Dx)
            for i in self.idx
        ])
        # apply the DtN matrix
        phi_vals = phi_func(self.x_vals)
        computed = (self.DtN_mat / self.Dx @ phi_vals)[self.idx]

        # relative L2 error
        rel_error = jnp.linalg.norm(computed - expected) \
                  / jnp.linalg.norm(expected)

        self.assertLessEqual(
            rel_error, 0.05,
            f"Relative L2 error {rel_error:.3%} exceeds 5%"
        )

    def test_sin_exp_decay(self):
        """φ(x)=sin(x)·exp(−(a x)^2)"""
        phi = lambda x: jnp.sin(x) * jnp.exp(-(self.a*x)**2)
        self.run_relative_test(phi)

    def test_cos_exp_decay(self):
        """φ(x)=cos(x)·exp(−(a x))"""
        phi = lambda x: jnp.cos(x) * jnp.exp(-(self.a*x))
        self.run_relative_test(phi)

    def test_plain_sin(self):
        """φ(x)=sin(x)"""
        phi = lambda x: jnp.sin(x)
        self.run_relative_test(phi)


if __name__ == "__main__":
    unittest.main()
