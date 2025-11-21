import jax.numpy as jnp
import unittest

from surferbot.rigid import rigidSolver

class RankTests(unittest.TestCase):
    @staticmethod
    def test_general_matrix_rank():
        [phi, eta, zeta, theta, A] = rigidSolver(1000, 10, 1e-6, 9.81, 0.05, 1, 0.072, 0.02, 1, 21)
        assert jnp.linalg.matrix_rank(A) == min(A.shape)
    @staticmethod    
    def test_bigger_raft_rank():
        [phi, eta, zeta, theta, A] = rigidSolver(1000, 10, 1e-6, 9.81, 0.3, 1, 0.072, 0.02, 1, 21)
        assert jnp.linalg.matrix_rank(A) == min(A.shape)

