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

    # Equation 4: Thrust
    thrust_factor = nu * rho**2 * L_raft/4 # TODO: This assumes drag for a flat plate as in blasius. 

    ## Helper variables
    L_raft_adim = L_raft / L_c
    L_domain_adim = jnp.floor(L_domain / L_c) - jnp.floor(L_domain/L_c) % 2 + 1 # We make it odd
    N = int(n * L_domain_adim / L_raft_adim)
    
    x = jnp.linspace(-L_domain_adim/2, L_domain_adim/2, N)
    x_contact = abs(x) <= L_raft_adim/2; H = sum(x_contact)
    x_free    = abs(x) >  L_raft_adim/2
    contact_boundary = jnp.array([(N-H)/2, (N+H)/2], dtype=int); 
    dx = (x[contact_boundary[0]] - x[contact_boundary[0]-1]).item(0) # TODO: Check if i want this to be a float
    contact_boundary2= jnp.array([(N-H)/2 - 1, (N+H)/2 + 1], dtype=int)
    M = 10 # Number of points in the z direction
    z = jnp.logspace(0, jnp.log10(2), M) - 1; #z[-1] = -1.

    # Define second derivative operators along x and z
    d_dx = Diff(axis=0, grid=jnp.round(x, 3), acc=2) # TODO: Adapt this to nonuniform grids
    d_dz = Diff(axis=1, grid=jnp.round(z, 3), acc=2)

    # Define the position of the motor
    x_motor = abs(x[x_contact] - motor_position/L_c) < 0.05 #TODO: Check that the motor occupying 5% of the raft is reasonable
    weights = 1. / sum(x_motor)
    
    Id = 1.0 # Identity operator

    ## BUILDING THE MATRIX SYSTEM
    # A little note on notation: The matrices A are fourth order tensors. 
    # The first two indices are the equations, and the last two are the variables.
    # For example, A[i, j, :, :] are the weights for the linear equation that corresponds to the node (i, j)

    # Building the first block of equations (Bernoullli on the surface)
    E1 = C11 * d_dz - C12 * d_dz * d_dx**2 - C13 * Id - C14 * d_dx**2
    E1 = E1.matrix((N, M)).toarray().reshape(N, M, N, M)
    E1 = E1[:, [0], :, :] # Only take the surface equations
    pad = jnp.zeros((N, 1, N, 1), dtype=jnp.complex64)
    E1 = jnp.concatenate([pad, E1], axis=3) # Adding surface variable
    # Only the equations that are not in contact. This shape should be (N-H, 1, N, M+1)
    E1 = E1[x_free, :, :, :].reshape(-1, N, M+1)  
    assert E1.shape[0] == N - H, f"Shape of E1 is {E1.shape} instead of {N-H}" if DEBUG else None
    
    # Building the second block of equations (Euler beam Equation)
    d_dxc = Diff(axis=0, grid=jnp.round(x[x_contact], 3), acc=2) # I dont want to take information from outside for the beam equation
    E2_eta = C21 * d_dxc**4 - C22 * Id - C25 * Id ## TODO: Change this to second order derivative
    E2_eta = E2_eta.matrix((H, 1)).toarray().reshape(H, 1, H, 1)
    aux = jnp.zeros((H, 1, N, 1), dtype=jnp.complex64)
    E2_eta = aux.at[:, :, x_contact, :].set(E2_eta)
    surface_tension = jnp.zeros((H, 1, N, 1), dtype=jnp.complex64)
    surface_tension = surface_tension.at[[0, -1], 0, contact_boundary, 0].set(-dx)
    surface_tension = surface_tension.at[[0, -1], 0, contact_boundary2, 0].set(dx)
    surface_tension = - C27 * surface_tension
    E2_eta = E2_eta + surface_tension

    E2_phi =  - C24 * Id - C26 * d_dx ** 2
    E2_phi = E2_phi.matrix((N, M)).toarray().reshape(N, M, N, M)
    E2_phi = E2_phi[x_contact, 0:1, :, :] # Only take the surface equations
    # This shape should be (H, 1, N, M+1)    
    E2 = jnp.concatenate([E2_eta, E2_phi], axis=3).reshape(-1, N, M+1) 
    assert E2.shape[0] == H, f"Shape of E2 is {E2.shape} instead of {H}" if DEBUG else None

    # Kinematic boundary conditions
    E3_phi = C31 * d_dz
    E3_phi = E3_phi.matrix((N, M)).toarray().reshape(N, M, N, M)
    E3_phi = E3_phi[:, [0], :, :] # Only take the free surface equations
    E3_eta = - C32 * jnp.eye(N).reshape(N, 1, N, 1)
    # This shape should be (N, 1, N, M+1)
    E3 = jnp.concatenate([E3_eta, E3_phi], axis=3).reshape(-1, N, M+1) 
    assert E3.shape[0] == N, f"Shape of E3 is {E3.shape} instead of {N}" if DEBUG else None

    # Laplacian
    E4 = d_dx**2 + d_dz**2
    E4 = E4.matrix((N, M)).toarray().reshape(N, M, N, M)
    # We remove the bottom of 
    # Adding surface variable. This shape should be (N, M, N, M+1)
    E4 = jnp.concatenate([jnp.zeros((N, M, N, 1)), E4], axis=3).reshape(-1, N, M+1) 
    assert E4.shape[0] == N*M, f"Shape of E4 is {E4.shape} instead of {N*M}" if DEBUG else None

    # Boundary condition at the bottom of the surface
    E5 = d_dz.matrix((N, M)).toarray().reshape(N, M, N, M)
    E5 = E5[:, [-1], :, :] 
    E5 = jnp.concatenate([pad, E5], axis=3).reshape(-1, N, M+1)
    assert E5.shape[0] == N, f"Shape of E5 is {E5.shape} instead of {N}" if DEBUG else None

    # Concatenate everything
    A = jnp.concatenate([E1, E2, E3, E4, E5], axis = 0)

    ## BUILDING THE INDEPENDENT COMPONENT
    b1 = jnp.zeros((E1.shape[0], ), dtype=jnp.complex64)
    # Weighting only where motor is applied
    b2 = - C23 * jnp.where(jnp.array(x_motor), weights, 0.0)
    b3 = jnp.zeros((E3.shape[0], ), dtype=jnp.complex64)
    b4 = jnp.zeros((E4.shape[0], ), dtype=jnp.complex64)
    b5 = jnp.zeros((E5.shape[0], ), dtype=jnp.complex64)
    b = jnp.concatenate([b1, b2, b3, b4, b5], axis = 0)

    ## SOLVING THE SYSTEM AND CALCULATING THRUST
    x = solve_tensor_system(A, b)

    eta = x[:, 0]
    phi = x[:, 1:]; phi_surface = phi[:, 0]

    assert test_solution(eta, phi_surface, (x, z)), "The solution does not satisfy the PDE" if DEBUG else None
    
    ## ASSEMBLE PRESSURE
    # Phi component of pressure
    P1 = (C24 * Id + C26 * d_dx ** 2).matrix((N, )).toarray().reshape(N, N)
    P1 = P1 @ phi_surface
    p = (C25 * eta + P1)[x_contact] # Pressure (eqn 2.20)

    Dx = d_dxc.matrix((H, )).toarray().reshape(H, H)
    eta_x = Dx @ eta[x_contact]

    integral = simpson_weights(H, dx)
    thrust = integral @ (-1/2 * (jnp.real(p) * jnp.real(eta_x) + jnp.imag(p) * jnp.imag(eta_x)))

    thrust = F_c * thrust # Adding dimensions
    U = jnp.power(thrust**2/thrust_factor , 1/3)
    return U


if __name__ == "__main__":
    A = solver()
