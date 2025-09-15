import jax.numpy as jnp
from surferbot.DtN import DtN_generator
import scipy.integrate as spi
from surferbot.myDiff import Diff
from surferbot.utils import solve_tensor_system, gaussian_load, test_solution, dispersion_k
from surferbot.integration import simpson_weights
import unittest

def solver(rho, omega, nu, g, L_raft, L_domain, gamma, n = 100, theta, zeta):
    '''
    Inputs: 
    - rho: density of fluid
    - omega: frequency
    - g: gravity (m/s^2)
    - L_raft: length of raft (m)
    - L_domain: length of domain (m)
    - motor_inertia: intertia of motor
    - nu: kinematic viscosity
    - gamma: surface tension
    - n: number of points in raft
    - N: total number of points x-direction
    - theta, zeta: treating as givens for now
    Outputs:
    - 
    '''

    ## Derived dimensional Parameters (SI Units)
    L_c = L_raft
    m_c = rho * L_c**2
    t_c = 1/omega
    F_c = rho * L_c**3 / t_c**2

    # Equation setup
    DtN = DtN_generator(n)
    N = DtN / (L_raft / n)
    integral = simpson_weights(H, dx)

    # Equation 1 (B)
    C11 = 1.0
    C12 = -(gamma / (rho * L_c**3 * omega**2))/(g / (L_c * omega**2))
    C13 = -(omega**2 * L_c / g)        
    C14 = -(4.0j * nu / (omega * L_c**2)) / (g / (L_c * omega**2))

    # Equation 2 (C)
    C21 = 1.0
    C22 = -1j * (zeta + theta * x) * omega  # constant term

    # Equation 3 (G)
    C31 = 1.0j
    C32 = -2 * nu / (omega * L_c**2)
    C33 = -1.0

    # Equation 4 (2.11b)
    C41 = -m_c
    C42 = -F_c  # constant term
    C43 = 1.0j
    C44 = g / (omega**2 * L_c)
    C45 = 2 * nu / (omega * L_c**2)

    # Equation 5 (2.11c)
    C51 = -(1/12) * m_c
    C52 = -(x / L_c) * F_c  # constant term
    C53 = 1.0j * x / L_c
    C54 = (g / (omega**2 * L_c)) * (x / L_c)
    C55 = (2 * nu / (omega * L_c**2)) * (x / L_c)

    # Raft Points
    L_raft_adim = L_raft / L_c
    L_domain_adim = jnp.floor(L_domain / L_c) - jnp.floor(L_domain/L_c) % 2 + 1 # We make it odd (for simson integration)
    N = jnp.int32(n * L_domain_adim / L_raft_adim) 

    if jnp.std(x) < 1e-5:
        grid_x = (x[left_raft_boundary] - x[left_raft_boundary-1]).item(0)
    else:
        grid_x = jnp.round(x, 5)
        
    x = jnp.linspace(-L_domain_adim/2, L_domain_adim/2, N)
    x_contact = abs(x) <= L_raft_adim/2; H = sum(x_contact)
    x_free    = abs(x) >  L_raft_adim/2; x_free = x_free.at[0].set(False); x_free = x_free.at[-1].set(False)
    assert jnp.sum(x_free) + jnp.sum(x_contact) == N-2, f"Number of free and contact points do not match the total number of points {N}." if DEBUG else None
    left_raft_boundary = (N-H)//2; right_raft_boundary = (N+H)//2

    # derivative operators
    dx = (x[left_raft_boundary] - x[left_raft_boundary-1]).item(0)
    d_dx = Diff(axis=0, grid=grid_x, acc=2, shape=(N, M))

    x_contact = x[abs(x) <= L_c]
    x_free    = x[abs(x) >  L_c]

    # Building matrix
    # [E11][0][0][0]      [phi]
    # [E21][0][0][0]      [eta]
    # [E31][E32][0][0]    [zeta]
    # [E41][E42][E43][0]  [theta]
    # [E51][E52][0][E54]    

    E11 = C11 + (C12 + C14) * d_dx**2 + C13
    E12 = 0
    E13 = 0
    E14 = 0

    E21 = C21 
    E22 = 0
    E23 = 0
    E24 = 0
    
    E31 = C33
    E32 = C31 + C32 * d_dx**2
    E33 = 0
    E34 = 0

    E41 = integral @ (C43 + C45 * d_dx**2) # (set endpoint?)
    E42 = C44
    E43 = C41
    E44 = 0

    E51 = integral @ (C53 + C55 * d_dx**2) 
    E52 = C54
    E53 = 0
    E54 = C51

    E1 = jnp.stack(E11, E12, E13, E14, axis = 1)
    E2 = jnp.stack(E21, E22, E23, E24, axis = 1)
    E3 = jnp.stack(E31, E32, E33, E34, axis = 1)
    E4 = jnp.stack(E41, E42, E43, E44, axis = 1)
    E5 = jnp.stack(E51, E52, E53, E54, axis = 1)

    # Concatenate (TODO: add in constant in E2, E4, E5)
    E = jnp.stack(E1, E2, E3, E4, E5, axis = 0)

    # TODO: add test checking E is a square 

    return A

if __name__ == "__main__":
    solver(1, 1, 1, 1, 100, 9.8, 10, 100)

