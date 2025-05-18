import jax.numpy as jnp
import jax
import unittest
from DtN import DtN_generator
import scipy.integrate as spi
from findiff import Diff
from integration import simpson_weights
 

def jDiff(axis, grid=None, periodic = False, acc=2):
    # If coords is jax array, convert it to pure float
    if isinstance(grid, jnp.ndarray):
        grid = jnp.round(grid, 5)
        grid = grid.tolist()
    return Diff(axis, grid=grid, periodic=periodic, acc=acc)




def solver(sigma, rho, omega, nu, eta, g, 
           L_raft, force, L_domain, E, I, rho_raft, n = 101):
    """
    Solves the linearized water wave problem for a flexible raft
    Inputs:
    - sigma: surface tension (N)
    - rho: density of water  (kg/m^2)
    - omega: angular frequency (rad/s)
    - nu: kinematic viscosity (m^2/s)
    - eta: wave amplitude (m)
    - g: gravitational acceleration (m/s^2)
    - L_raft: length of the raft (m)
    - force: force applied to the raft (N)
    - mass: mass of the raft (kg)
    - L_domain: length of the domain (m)
    - E: Young's modulus (Pa)
    - I: moment of inertia (m^4)
    - rho_raft: density of the raft (per unit length) (kg/m)
    - n: number of points in the raft
    Outputs:
    - U: Terminal Velocity of the raft
    
    There are three equations to be solved. 
    """
    ## Derived dimensional Parameters (SI Units)
    L_c = L_raft
    t_c = 1/omega
    m_c = rho_raft * L_c
    F_c = m_c * L_c**2 / t_c


    ## Non-dimensional Parameters
    ## Equation 1: Bernoulli on free surface
    C11 = 1.0
    C12 = sigma/(rho * g * L_c**2)
    C13 = omega**2 / g * L_c                
    C14 = 4.0j * nu * omega / (g*L_c)       # Viscous drag
    ## Equation 2: Euler Beam equation
    C21 = E*I/(m_c * L_c**3 / t_c**2)       # Elasticity
    C22 = 1.
    C23 = force / (m_c * L_c / t_c**2)      # Force
    C24 = - rho/rho_raft * 1.0j * L_c       # Inertia
    C25 = - rho /rho_raft * g * t_c**2      # Gravity
    C26 = rho/rho_raft * 2 * nu * t_c / L_c # Viscous drag
    C27 = sigma / (rho_raft * L_c / t_c**2) # Surface tension force

    ## Equation 3: Kinematic boundary condition
    C31 = 1.
    C32 = 1.0j

    # Equation 4: Harmonic eqn (fluid bulk)
    # No coefficients here here

    ## Helper variables
    L_raft_adim = L_raft / L_c
    L_domain_adim = jnp.floor(L_domain / L_c) - jnp.floor(L_domain/L_c) % 2 + 1 # We make it odd
    N = n * L_domain_adim / L_raft_adim
    
    x = jnp.linspace(-L_domain_adim, L_domain_adim, N)
    x_contact = abs(x) <= L_raft_adim/2; H = sum(x_contact)
    x_free    = abs(x) >  L_raft_adim/2
    contact_boundary = x_contact == True & (x_contact[1:] == False | x_contact[:-1] == False)
    contact_boundary2= x_contact == False & (x_contact[1:] == True | x_contact[:-1] == True)
    h = x[1] - x[0]
    M = N
    z = jnp.logspace(0, jnp.log10(2), M) - 1; z[-1] = -1.

    # Define second derivative operators along x and z
    d_dx = Diff(axis=0, grid=jnp.round(x, 3).to_list(), acc=2)
    d_dz = Diff(axis=1, grid=jnp.round(z, 3).to_list(), acc=2)
    
    Id = 1.0 + d_dx - d_dx # Dummy identity operator

    ## BUILDING THE MATRIX SYSTEM
    # Building the first block of equations (Bernoullli on the surface)
    E1 = C11 * d_dz - C12 * d_dz * d_dx**2 - C13 * Id - C14 * d_dx**2
    E1 = E1.matrix((N, M)).toarray().reshape(N, M, N, M)
    E1 = E1[:, 0, :, :] # Only take the surface equations
    pad = jnp.zeros((N, 1, N, 1), dtype=jnp.complex64)
    E1 = jnp.concatenate([pad, E1], axis=3) # Adding surface variable
    # Only the equations that are not in contact. This shape should be (N-H, 1, N, M+1)
    E1 = E1[x_free, :, :, :].reshape(-1, N, M+1)  

    # Building the second block of equations (Euler beam Equation)
    d_dx_contact = Diff(axis=0, grid=jnp.round(x[x_contact], 3).to_list(), acc=2) # I dont want to take information from outside for the beam equation
    E2_eta = C21 * d_dx_contact ** 4 - C22 * Id - C25 * Id  
    E2_eta = E2_eta.matrix((H, )).toarray().reshape(H, 1, H, 1)
    aux = jnp.zeros((H, 1, N, 1), dtype=jnp.complex64)
    aux = aux.at[:, :, x_contact, :].set(E2_eta)
    E2_eta = aux
    surface_tension = jnp.zeros((H, 1, N, 1), dtype=jnp.complex64)
    surface_tension = surface_tension.at[contact_boundary, 0, contact_boundary, 0].set(-1.)
    surface_tension = surface_tension.at[contact_boundary2, 0, contact_boundary2, 0].set(1.)
    surface_tension = - C27 * surface_tension
    E2_eta = E2_eta + surface_tension

    E2_phi =  - C24 * Id - C26 * d_dx ** 2
    E2_phi = E2_phi.matrix((N, M)).toarray().reshape(N, M, N, M)
    E2_phi = E2_phi[x_contact, 0, :, :] # Only take the surface equations

    # This shape should be (H, 1, N, M+1)    
    E2 = jnp.concatenate([E2_eta, E2_phi], axis=3).reshape(-1, N, M+1) 


    # Kinematic boundary conditions
    E3_phi = C31 * d_dz
    E3_eta = - C32 * Id
    E3_phi = E3_phi.matrix((N, M)).toarray().reshape(N, M, N, M)
    E3_phi = E3_phi[:, 0, :, :] # Only take the free surface equations
    E3_eta = E3_eta.matrix((N, )).toarray().reshape(N, 1, N, 1)
    # This shape should be (N, 1, N, M+1)
    E3 = jnp.concatenate([E3_eta, E3_phi], axis=3).reshape(-1, N, M+1) 

    # Laplacian
    E4 = d_dx**2 + d_dz**2
    E4 = E4.matrix((N, M)).toarray().reshape(N, M, N, M)
    # Adding surface variable. This shape should be (N, M, N, M+1)
    E4 = jnp.concatenate([pad, E4], axis=3).reshape(-1, N, M+1) 

    # Concatenate everything
    A = jnp.concatenate([E1, E2, E3, E4], axis = 0)

    ## BUILDING THE INDEPENDENT COMPONENT


    return A

def solve_tensor_system(A, b):
    m = A.shape[0]
    x_shape = A.shape[1:]
    A_mat = A.reshape(m, -1)  # shape (m, N)
    b_vec = b.reshape(m)
    if m == b.shape[0]:
        x_flat = jnp.linalg.solve(A_mat, b_vec)
    else:
        x_flat, _, _, _ = jnp.linalg.lstsq(A_mat, b_vec, rcond=None)

    return x_flat.reshape(x_shape)

if __name__ == "__main__":
    solver(1, 1, 1, 1, 100, 9.8, 10, 100)

