import jax.numpy as jnp
import unittest
from DtN import DtN_generator
import scipy.integrate as spi
from findiff import Diff
 
# constants
x = jnp.linspace(0, 10, 100)
h = x[1] - x[0]
n = 12 # size of DtN
g = 9.8
# delta = 
# rho = 
# omega = 
# nu = 
# phi = 


def solver(sigma, rho, omega, nu, n, g, L, domain):
    
    DtN = DtN_generator(n)
    N = DtN / (L / n)

    # findiff
    first_deriv = Diff(0, h).matrix((n, ))
    d_dx = jnp.array(first_deriv.toarray())

    d2_dx2 = Diff(0, float(h), acc=2) ** 2

    # manual finite differences
    C = (4 * 1j * nu  * omega / g)
    
    A = N @ d_dx(phi_vector) - N * (sigma / (rho * g)) @ d2_dx2(phi_vector) - (omega**2 / g) * d_dx - C * d2_dx2

    return A

if __name__ == "__main__":
    solver(1, 1, 1, 1, 100, 9.8, 10, 100)

