import jax.numpy as jnp
import numpy as np
import pytest

from surferbot.rigid_surferbot import rigidSolver

# testing variables
rho = 1000.
omega = 2*jnp.pi*80.
nu = nu = 1e-6
g = 9.81
L_raft = 0.05
L_domain = 1.
gamma = 72.2e-3
x_A = 0.025 # check this
f_A = 1. # check this
n = 100

# def general_test(rho, omega, nu, g, L_raft, L_domain, gamma, x_A, f_A, n):
#     [phi, theta, eta, zeta, r, c] = rigidSolver(rho, omega, nu, g, L_raft, L_domain,gamma, x_A, f_A, n)
    
#     assert phi == 
#     assert theta ==
#     assert eta ==
#     assert zeta == 
def test_square_matrix_test(rho, omega, nu, g, L_raft, L_domain, gamma, x_A, F_A, n):
    [phi, theta, eta, zeta, r, c] = rigidSolver(rho, omega, nu, g, L_raft, L_domain,gamma, x_A, f_A, n)
    assert r == c
