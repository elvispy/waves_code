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


def surferbot(sigma, rho, omega, nu, g, L_raft, L_domain, n: int = 100):

    ## Derived dimensional Parameters (SI Units)


    ## Non-dimensional Parameters


    ## Helper variables
    x = jnp.linspace(-L_domain, L_domain, n)
    x_contact = x[abs(x) <= L_raft]
    x_free    = x[abs(x) >  L_raft]
    h = x[1] - x[0]
    DtN = DtN_generator(n)
    N = DtN / (L_raft / n)

    # findiff
    first_deriv = Diff(0, h).matrix((n, ))
    d_dx = jnp.array(first_deriv.toarray())

    d2_dx2 = Diff(0, float(h), acc=2) ** 2


    ## Building the first block of equations (Bernoullli on free surface)
    # manual finite differences
    C = (4 * 1j * nu  * omega / g)

    A = N @ d_dx(phi_vector) - N * (sigma / (rho * g)) @ d2_dx2(phi_vector) - (omega**2 / g) * d_dx - C * d2_dx2
    ## Building the second block of equations (Kinematic BC on the free surface)

    ## Building the third block of equations (Kinematic BC on the raft)

    ## Building the fourth block (Raft-free surface BC)

    ## 

    return A

if __name__ == "__main__":
    surferbot(1, 1, 1, 1, 100, 9.8, 10, 100)

