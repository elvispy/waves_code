from copy import deepcopy
import jax.numpy as jnp
from surferbot.constants import DEBUG
from surferbot.myDiff import Diff
from surferbot.integration import simpson_weights
from surferbot.utils import solve_tensor_system, gaussian_load, test_solution
from surferbot.sparse_utils import _SparseAtProxy
import jax.experimental.sparse as jsparse
# Adding the add and set properties do BCOO
jsparse.BCOO.at = property(lambda self: _SparseAtProxy(self))

def solver(sigma = 72.20, rho = 30., omega = 2*jnp.pi*80., nu = 1e-6, g = 9.81, 
           L_raft = 0.05, force = 2.74, motor_position = 1.9/5 * 0.05, 
           L_domain = .5, EI = 3.0e+9 * 3e-2 * 1e-4**3 / 12, # 3GPa times 3 cm times 0.05 cm ^4 / 12
           rho_raft = 0.018 * 3., n = 21, M = 10): #TODO CHECK UNITS FOR THE PROBLEM TO BE OF DEPTH 3cm (original surferbot)
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
    - EI: Young's modulus (Pa) * moment of inertia (m^4)
    - rho_raft: density of the raft (per unit length) (kg/m)
    - n: number of points in the raft
    - M: number of points in the z direction
    - DEBUG: whether to print debug information
    Outputs:
    - U: Terminal Velocity of the raft
    
    There are three equations to be solved. 
    """
    ## Derived dimensional Parameters (SI Units)
    L_c = L_raft
    t_c = 1/omega
    m_c = rho * L_c**2
    F_c = m_c * L_c**2 / t_c

    ## Non-dimensional Parameters
    ## Equation 1: Bernoulli on free surface
    C11 = 1.0
    C12 = -sigma/(rho * g * L_c**2)
    C13 = -omega**2 / g * L_c                
    C14 = -4.0j * nu * omega / (g*L_c)       # Viscous drag

    ## Equation 2: Euler Beam equation
    C21 = EI * t_c /(1.0j * omega * m_c * L_c**3 )        # Elasticity
    C22 = -rho_raft * omega * L_c * t_c / (1.0j * m_c)    # Inertia
    C23 = -force / (m_c * L_c / t_c**2)                   # Force load
    C24 = (rho * t_c * 1.0j * omega * L_c**2) / m_c       # Inertia
    C25 = (rho * t_c * g * L_c) / (m_c * 1.0j * omega)    # Gravity
    C26 = -(rho * t_c * 2 * nu) / m_c                     # Viscous drag
    C27 = -sigma * t_c / (1.0j * omega * m_c * L_c)       # Surface tension force

    ## Equation 3: Radiative boundary conditions
    k = 1.0 #TODO: Calculate the dispersion relation
    C31 = 1.
    C32 = 1.0j * k * L_c

    # Equation 4: Harmonic eqn (fluid bulk)
    # No coefficients here here

    # Equation 5: Thrust
    thrust_factor = 4/9 * nu * rho**2 * L_raft 

    ## Helper variables
    L_raft_adim = L_raft / L_c
    L_domain_adim = jnp.floor(L_domain / L_c) - jnp.floor(L_domain/L_c) % 2 + 1 # We make it odd
    N = jnp.int32(n * L_domain_adim / L_raft_adim) 
        
    x = jnp.linspace(-L_domain_adim/2, L_domain_adim/2, N)
    x_contact = abs(x) <= L_raft_adim/2; H = sum(x_contact)
    x_free    = abs(x) >  L_raft_adim/2; x_free = x_free.at[0].set(False); x_free = x_free.at[-1].set(False) # TODO: Check if this is correct
    left_raft_boundary = (N-H)//2; right_raft_boundary = (N+H)//2
    dx = (x[left_raft_boundary] - x[left_raft_boundary-1]).item(0) # TODO: Check if i want this to be a float
    #contact_boundary2= jnp.array([(N-H)/2 - 1, (N+H)/2 + 1], dtype=int)
    z = jnp.logspace(0, jnp.log10(2), M) - 1; #z[-1] = -1.

    # Define second derivative operators along x and z
    d_dx = Diff(axis=0, grid=jnp.round(x, 3), acc=2, shape=(N, M)) # TODO: Adapt this to nonuniform grids
    d_dz = Diff(axis=1, grid=jnp.round(z, 3), acc=2, shape=(N, M))

    # Define the position of the motor
    weights = gaussian_load(motor_position/L_c, 0.01, x[x_contact])
    
    I_NM = jnp.eye(N)[:, None, :, None] * jnp.eye(M)[None, :, None, :]
    O_NM = jnp.zeros((N, N), dtype=jnp.complex64)[:, None, :, None] * jnp.zeros((M, M), dtype=jnp.complex64)[None, :, None, :]

    ## BUILDING THE MATRIX SYSTEM
    # A little note on notation: We want to construct a fifth order tensor of size (N, M, N, M, 5) 
    # The first two indices are the equations, and the last three are the variables.
    # For example, A[i, j, :, :, :] are the weights for the linear equation that corresponds to the node (i, j)
    # and A[:, :, i, j, k] are the weights for the variable \partial_x^k \phi_{i, j} at the node (i, j)
    # That is to say, the k index is the order of the derivative.

    # Building the first block of equations (Bernoullli on the surface)
    E1 = jnp.stack([C11 * d_dz + C13 * I_NM, O_NM, C12 * d_dz + C14 * I_NM, O_NM, O_NM], axis = -1)

    # Only the equations that are not in contact. This shape should be (N-H, 1, N, M+1)
    E1 = E1.at[~x_free, 0, :, :, :].set(0.0) # Remove the contact equations at the surface
    E1 = E1.at[:      ,1:, :, :, :].set(0.0) # Just to make sure equations that do not refer to the surface dont enter
    assert E1.shape == (N, M, N, M, 5), f"Shape of E1 is {E1.shape}." if DEBUG else None 
    
    # Building the second block of equations (Euler beam Equation)
    d_dx1D = Diff(axis=0, grid=jnp.round(x[x_contact], 3), acc=2, shape=(H, M)) # I dont want to take information from outside for the beam equation
    d_dz1D = deepcopy(d_dz); d_dz1D.shape = (H, M); d_dz1D = 1.0 * d_dz1D
    I_HM = jnp.eye(H)[:, None, :, None] * jnp.eye(M)[None, :, None, :]
    O_HM = jnp.zeros(I_HM.shape, dtype=jnp.complex64)

    E12 = jnp.stack([C22 * d_dz1D + C24 * I_HM + C25 * d_dz1D, O_HM, C26 * I_HM, O_HM, C21 * d_dz1D], axis = -1)
    E12 = E12.at[:, 1:, :, :, :].set(0.0) # Just to make sure equations that do not refer to the surface dont enter
    
    # Bring both idx arrays into the same index-tuple
    idx = jnp.nonzero(x_contact)[0]
    E1 = E1.at[idx[:, None], :, idx[None, :], :, :].set(jnp.swapaxes(E12, 1, 2))
    
    d_dz_surface = deepcopy(d_dz); d_dz_surface.shape = (sum(x_free)//2, M)
    # Take the derivative at the surface (no inside information)
    d_dx_free    = Diff(0, grid=x[jnp.logical_and(x_free, x > 0)], acc=2, shape=(sum(x_free)//2, M)) #.matrix((len(x[jnp.logical_and(x_free, x > 0)]), )).toarray()[0, :]

    d_dxdz = ((d_dx_free * d_dz_surface).op())[0, 0, :, :]
    # Surface tension term on the right
    E1 = E1.at[right_raft_boundary, 0, jnp.logical_and(x_free, x > 0), :, 1].add( C27/dx * d_dxdz)
    E1 = E1.at[left_raft_boundary , 0, jnp.logical_and(x_free, x < 0), :, 1].add(-C27/dx * jnp.flip(d_dxdz, axis=0))

    # Laplacian in operator form
    E2 = (1.0 * d_dx**2 + 1.0 * d_dz**2)

    # We remove the bottom of 
    # Adding surface variable. This shape should be (N, M, N, M+1)
    E2 = E2.at[[0, -1], :, :, :].set(0.0) # Remove the boundaries
    E2 = E2.at[:, [0, -1], :, :].set(0.0) # Remove the boundaries
    E2 = jnp.stack([E2, O_NM, O_NM, O_NM, O_NM], axis=-1)
    assert E2.shape == (N, M, N, M, 5), f"Shape of E4 is {E2.shape} instead of {N*M}" if DEBUG else None

    # Boundary condition at the bottom of the surface
    E3 = 1.0 * d_dz
    E3 = E3.at[:, :-1, :, :].set(0.0)
    E3 = E3.at[[0, -1], :, :, :].set(0.0)
    E3 = jnp.stack([E3, O_NM, O_NM, O_NM, O_NM], axis=-1)
    #assert E5.shape[0] == N, f"Shape of E5 is {E5.shape} instead of {N}" if DEBUG else None

    # Radiative boundary conditions
    E4 = jnp.zeros((N, M, N, M, 5), dtype=jnp.complex64)
    E4 = E4.at[0 , :, 0, :, 0].set(C32) # Leftmost equations, left coefficients, zeroth derivative coefficients
    E4 = E4.at[0 , :, 0, :, 1].set(C31) # Leftmost equations, left coefficients, first derivative coefficients
    E4 = E4.at[-1, :,-1, :, 0].set(-C32)# Rightmost equations, right coefficients, zeroth derivative coefficients
    E4 = E4.at[-1, :,-1, :, 1].set(C31) # Rightmost equations, right coefficients, first derivative coefficients
    # Concatenate everything
    E = E1 + E2 + E3 + E4

    # Horizontal derivatives definition
    Dx = 1.0 * d_dx
    A = jnp.stack([E,
                   jnp.stack([Dx, -I_NM, O_NM, O_NM, O_NM], axis = -1),
                   jnp.stack([O_NM, Dx, -I_NM, O_NM, O_NM], axis = -1),
                   jnp.stack([O_NM, O_NM, Dx, -I_NM, O_NM], axis = -1),
                   jnp.stack([O_NM, O_NM, O_NM, Dx, -I_NM], axis = -1)], axis = 0)

    assert A.shape == (5, N, M, N, M, 5), f"Shape of A is {A.shape} instead of {(5, N, M, N, M, 5)}" if DEBUG else None
    assert jnp.any(jnp.all(jnp.abs(A) < 1e-10, axis=(-3, -2, -1))) == False, "Matrix A is defficient" if DEBUG else None
    ## BUILDING THE INDEPENDENT COMPONENT
    b = jnp.zeros((5, N, M))
    # Only apply the force where the raft is in contact
    b = b.at[0, x_contact, 0].set(-C23 * weights) 

    ## SOLVING THE SYSTEM AND CALCULATING THRUST
    x = solve_tensor_system(A, b) #TODO Determinant is zero. Check for example A[0, 0, 1, :5, :5, 0]

    phi = x[:, :, 0]; phi_surface = phi[:, 0]
    # Extract eta from the kinematic boundary condition
    eta = ((1.0/(1.0j * omega * t_c) * d_dz) @ phi)[:, 0]

    assert test_solution(eta, phi_surface, (x, z)), "The solution does not satisfy the PDE" if DEBUG else None
    
    ## ASSEMBLE PRESSURE
    # Phi component of pressure
    d_dx1D.shape = (H,)
    P1 = (C24 + C26 * d_dx1D ** 2) # TODO: Check that i want to calculate pressure only with information from inside the raft
    P1 = P1 @ (phi_surface[x_contact])
    p = (C25 * eta[x_contact] + P1) # Pressure (eqn 2.20)

    eta_x = (1.0 * d_dx1D) @ eta[x_contact]

    integral = simpson_weights(H, dx) # TODO: This assumes uniform grid
    thrust = integral @ (-1/2 * (jnp.real(p) * jnp.real(eta_x) + jnp.imag(p) * jnp.imag(eta_x)))

    thrust = F_c * thrust # Adding dimensions #TODO: Review sign convention here + maybe x-direction and small oscillations cancel each other here
    thrust = thrust + sigma * (eta[left_raft_boundary] - eta[left_raft_boundary-1]) / (x[left_raft_boundary] - x[left_raft_boundary-1])
    thrust = thrust - sigma * (eta[right_raft_boundary+1] - eta[right_raft_boundary]) / (x[right_raft_boundary+1] - x[right_raft_boundary])                              
    U = jnp.power(thrust**2/thrust_factor , 1/3)
    return U


if __name__ == "__main__":
    #s = jax.jit(solver)
    A = solver()#s()
    print(A)
