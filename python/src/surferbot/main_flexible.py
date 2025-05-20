import jax.numpy as jnp
import jax
from findiff import Diff as _Diff
from findiff import grids
import numpy as np
import numbers
import types
from integration import simpson_weights
 
def make_axis(dim, config_or_axis, periodic=False):
    """
    A reimplementation of the make_axis function to create a grid axis.
    Supports Jax arrays
    """
    if isinstance(config_or_axis, grids.GridAxis):
        return config_or_axis
    if isinstance(config_or_axis, numbers.Number):
        return grids.EquidistantAxis(dim, spacing=config_or_axis, periodic=periodic)
    elif isinstance(config_or_axis, jnp.ndarray) and len(config_or_axis) == 1:
        return grids.EquidistantAxis(dim, spacing=config_or_axis.item(0), periodic=periodic)
    elif isinstance(config_or_axis, (np.ndarray, jnp.ndarray)):
        return grids.NonEquidistantAxis(dim, coords=config_or_axis, periodic=periodic)

def _dynamic_s_op_method(S, shape=None):
    target_shape = shape if shape is not None else S.shape
    if isinstance(target_shape, tuple) and len(target_shape) <= 2:
        return jnp.array(S.matrix(target_shape).toarray().reshape(*(target_shape + target_shape)))
    else:
        return NotImplemented

class Diff(_Diff):

    def __init__(self, axis=0, grid=None, shape= None, periodic=False, acc=_Diff.DEFAULT_ACC):
        grid_axis = make_axis(axis, grid, periodic)
        super().__init__(axis, grid_axis, acc)
        self.shape = shape

    # TODO: Write a test to check if this is AD compliant.
    def op(self, shape=None):
        return _dynamic_s_op_method(self, shape=shape)
        
    def __pow__(self, power):
        """Returns a Diff instance for a higher order derivative."""
        new_diff = Diff(self.dim, self.axis, shape=self.shape, acc=self.acc)
        new_diff._order *= power
        return new_diff

    def __mul__(self, other):
        if isinstance(other, Diff) and self.dim == other.dim:
            new_diff = Diff(self.dim, self.axis, shape=self.shape, acc=self.acc)
            new_diff._order += other.order
            return new_diff
        elif isinstance(other, (numbers.Number, jnp.ndarray)):
            return other * self.op()
        S = super().__mul__(other)
        if self.shape == other.shape:
            S.shape = self.shape
            S.op = types.MethodType(_dynamic_s_op_method, S)
            return S
        return NotImplemented
    
    def __rmul__(self, other):
        if isinstance(other, (numbers.Number, jnp.ndarray)):
            return self.__mul__(other)
        return super().__rmul__(other)
    


def solve_tensor_system(A, b):
    m = A.shape[0]
    x_shape = A.shape[1:]
    A_mat = A.reshape(m, -1)  # shape (m, N)
    b_vec = b.reshape(m)
    if A_mat.shape[0] == A_mat.shape[1]:
        # If the system is square, use solve
        x_flat = jnp.linalg.solve(A_mat, b_vec)
    else:
        x_flat, _, _, _ = jnp.linalg.lstsq(A_mat, b_vec, rcond=None)

    return x_flat.reshape(x_shape)

def test_solution(eta, phi, domain):
    """
    Test the solution by checking the continuity of the velocity field
    """
    # TODO: Implement this function (all below is GPT)
    return True
    x, z = domain
    N = x.shape[0]
    M = z.shape[0]
    
    # Calculate the velocity field
    u = jnp.zeros((N, M), dtype=jnp.complex64)
    v = jnp.zeros((N, M), dtype=jnp.complex64)
    
    # Calculate the velocity field using finite differences
    d_dx = Diff(axis=0, grid=x, acc=1)
    d_dz = Diff(axis=1, grid=z, acc=1)
    
    u = d_dx @ phi[:, 0]  # Velocity in x direction
    v = d_dz @ eta       # Velocity in z direction
    
    # Check continuity of the velocity field
    continuity_check = jnp.allclose(u + v, 0.0)
    
    return continuity_check

def solver(sigma = 72.20, rho = 1000., omega = 60., nu = 1e-5, g = 9.81, 
           L_raft = 0.1, force = 1., motor_position = 0.025, 
           L_domain = 1., E = 1., I = 1., 
           rho_raft = 1240., n = 21, DEBUG = True):
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
    N = int(n * L_domain_adim / L_raft_adim) # TODO: Jax does not like this for jitting
        
    x = jnp.linspace(-L_domain_adim/2, L_domain_adim/2, N)
    x_contact = abs(x) <= L_raft_adim/2; H = sum(x_contact)
    x_free    = abs(x) >  L_raft_adim/2
    contact_boundary = jnp.array([(N-H)/2, (N+H)/2], dtype=int); 
    dx = (x[contact_boundary[0]] - x[contact_boundary[0]-1]).item(0) # TODO: Check if i want this to be a float
    contact_boundary2= jnp.array([(N-H)/2 - 1, (N+H)/2 + 1], dtype=int)
    M = 10 # Number of points in the z direction
    z = jnp.logspace(0, jnp.log10(2), M) - 1; #z[-1] = -1.

    # Define second derivative operators along x and z
    d_dx = Diff(axis=0, grid=jnp.round(x, 3), acc=2, shape=(N, M)) # TODO: Adapt this to nonuniform grids
    d_dz = Diff(axis=1, grid=jnp.round(z, 3), acc=2, shape=(N, M))

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
