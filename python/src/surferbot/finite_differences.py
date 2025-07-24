import jax.numpy as jnp
import unittest
from DtN import DtN_generator
import scipy.integrate as spi
from surferbot.myDiff import Diff
from surferbot.utils import solve_tensor_system, gaussian_load, test_solution, dispersion_k
from integration import simpson_weights

# question: do i need to be worried about motor inertia?
def solver(rho, omega, nu, g, L_raft, L_domain, motor_inertia, gamma, n = 100):
    '''
    Inputs: 
    - 
    Outputs:
    - 
    '''

    ## Derived dimensional Parameters (SI Units)
    force = motor_inertia *  omega**2 # question: Unsure if this is right, also when to use t_c vs omega?
    L_c = L_raft

    m_c = rho * L_c**2
    t_c = 1/omega
    F_c = rho * L_c**3 / t_c**2

    # Equations
    # Equation 1    (2.18b)
    C11 = 1.0
    C12 = -(gamma * rho * L_c**3 * omega**2)/(rho * g * L_c * omega**2)
    C13 = -omega**2 / (g * omega **2 * L_c)             
    C14 = -4.0j * nu * omega * L_c**2 / (g * L_c * omega**2)

    # Equation 2    (2.18c)
    C21 = 1.0
    C22 = 

    # Used in E4
    k = dispersion_k(omega, g, D, nu, sigma, rho) 
    C31 = 1.
    C32 = 0.0 * 1.0j * k * L_c
    
    # Equation 4        (2.11b)

    # Equation 5        (2.11c)

    # Equation 6        (2.19)

    # Equation 7        (2.20)

    # Equation 8        (2.21)


    # Raft Points
    L_raft_adim = L_raft / L_c
    L_domain_adim = jnp.floor(L_domain / L_c) - jnp.floor(L_domain/L_c) % 2 + 1 # We make it odd (for simson integration)
    N = jnp.int32(n * L_domain_adim / L_raft_adim) 

    # question: why do you need if-else here
    if jnp.std(x) < 1e-5:
        grid_x = (x[left_raft_boundary] - x[left_raft_boundary-1]).item(0)
    else:
        grid_x = jnp.round(x, 5)
        
    x = jnp.linspace(-L_domain_adim/2, L_domain_adim/2, N)
    x_contact = abs(x) <= L_raft_adim/2; H = sum(x_contact)
    x_free    = abs(x) >  L_raft_adim/2; x_free = x_free.at[0].set(False); x_free = x_free.at[-1].set(False)
    assert jnp.sum(x_free) + jnp.sum(x_contact) == N-2, f"Number of free and contact points do not match the total number of points {N}." if DEBUG else None
    left_raft_boundary = (N-H)//2; right_raft_boundary = (N+H)//2

    dx = (x[left_raft_boundary] - x[left_raft_boundary-1]).item(0)
    d_dx = Diff(axis=0, grid=grid_x, acc=2, shape=(N, M))

    x_contact = x[abs(x) <= L_c]
    x_free    = x[abs(x) >  L_c]

    # question: unsure what's going on in the z-direction (and M)
    d_dz = Diff(axis=1, grid=grid_z, acc=2, shape=(N, M))

    integral = simpson_weights(H, dx)
    DtN = DtN_generator(n)
    N = DtN / (L_raft / n)


    # Building Matrix
    E2 = (d_dx**2 + d_dz**2)      # question: for z = 0? (2.18a)

    E3 = d_dz ;  E3[:, :-1] = 0     # (2.18e)

    E4 =  E4 = jnp.zeros((N, M, N, M, 5), dtype=jnp.complex64)     # (2.18d)
    E4 = E4.at[0 , :, 0, :, 0].set(-C32) 
    E4 = E4.at[0 , :, 0, :, 1].set(C31) 
    E4 = E4.at[-1, :,-1, :, 0].set(C32)
    E4 = E4.at[-1, :,-1, :, 1].set(C31)



    return A

if __name__ == "__main__":
    solver(1, 1, 1, 1, 100, 9.8, 10, 100)

