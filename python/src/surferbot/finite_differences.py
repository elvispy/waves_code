import jax.numpy as jnp
import unittest
from DtN import DtN_generator
import scipy.integrate as spi
from findiff import Diff
from integration import simpson_weights
 
# constants
x = jnp.linspace(0, 10, 100)
h = x[1] - x[0]
n = 12 # size of DtN
g = 9.8
# delta = 
# rho = 
# omega = 
# nu = 


def solver(sigma, rho, omega, nu, eta, zeta, theta, g, L_raft, force, mass, L_domain, n = 100):

    ## Derived dimensional Parameters (SI Units)


    ## Non-dimensional Parameters


    ## Helper variables
    '''
    x = jnp.linspace(-L_domain, L_domain, n)
    x_contact = x[abs(x) <= L_raft]
    x_free    = x[abs(x) >  L_raft]
    h = x[1] - x[0]
    '''
    

    DtN = DtN_generator(n)
    N = DtN / (L_raft / n)
    # findiff
    first_deriv = Diff(0, h).matrix((n, ))
    d_dx = jnp.array(first_deriv.toarray())

    d2_dx2 = Diff(0, float(h), acc=2) ** 2

    constant = (4 * 1j * nu  * omega / g)
    
    ## Building the first block of equations (Bernoullli on free surface)
    B = N @ d_dx - N * (sigma / (rho * g)) @ d2_dx2 - (omega**2 / g) * d_dx - constant * d2_dx2

    # fix eta, zeta, theta, currently not a matrix

    ## Building the second block of equations (Kinematic BC on the free surface)
    C = N @ d_dx - 1j(zeta + x * theta) * omega

    G = N @ d_dx - 1j * omega * eta @ d_dx + 2 * nu * eta @ d2_dx2

    H = eta - zeta + theta * L_raft / 2

    # fix function
    f: lambda x: 1j * omega @ d_dx + g * zeta + nu * N**2 @ d_dx
    w = simpson_weights(n, L_raft / (n-1))

    A1 = force + mass * omega**2 * zeta - rho * jnp.dot(w, f(x))



    

    ## Building the third block of equations (Kinematic BC on the raft)

    ## Building the fourth block (Raft-free surface BC)

    ## 

    return A

if __name__ == "__main__":
    solver(1, 1, 1, 1, 100, 9.8, 10, 100)

