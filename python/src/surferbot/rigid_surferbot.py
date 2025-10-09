import jax
import jax.numpy as jnp
from surferbot.DtN import DtN_generator
from surferbot.myDiff import Diff
from surferbot.integration import simpson_weights
from surferbot.constants import DEBUG


def rigidSolver(rho, omega, nu, g, L_raft, L_domain, gamma, x_A, F_A, n):
    '''
    Inputs: 
    - rho: density of fluid
    - omega: frequency
    - g: gravity (m/s^2)
    - L_raft: length of raft (m)
    - L_domain: length of domain (m)
    - motor_inertia: intertia of motor
    - nu: kinematic viscosity
    - gamma: surface tension (sigma is gamma in the other notation)
    - n: number of points in raft
    - p: total number of points in domain (x-direction)
    Outputs:
    - 
    '''

    ## Derived dimensional Parameters (SI Units)
    # TODO: Fix these to be dimensional? - > fix wherever you use them as well
    L_c = L_raft
    m_c = rho * L_c**2
    t_c = 1/omega

    # Equation setup
    DtN = DtN_generator(n)
    dx = 1/n
    N = DtN / (L_raft / n) # operator
    print(len(N))
    integral = simpson_weights(n, dx)

    # Equation 1 (Bernoulli equation)
    C11 = 1.0
    C12 = -(gamma / (rho * L_c**3 * omega**2))/(g / (L_c * omega**2))
    C13 = -(omega**2 * L_c / g)      
    C14 = -(4.0j * nu / (omega * L_c**2)) / (g / (L_c * omega**2))

    # Kinematic boundary conditions
    # Equation 2 (inside the raft)
    C21 = 1.0
    C22 = -1j
    C23 = -1j 

    # Equation 3 (outside the raft)
    C31 = 1.0j
    C32 = -2 * nu / (omega * L_c**2)
    C33 = -1.0

    # Newton equations
    # Equation 4
    C41 = -m_c
    C42 = -F_A / (rho * L_c**3 * omega**2) # constant term
    C43 = 1.0j
    C44 = g / (omega**2 * L_c)
    C45 = 2 * nu / (omega * L_c**2)

    # Equation 5 
    C51 = -(1/12) * m_c
    C52 = -(x_A / L_c) * (F_A / (rho * L_c**3 * omega**2)) # constant term
    C53 = 1.0j
    C54 = (g / (omega**2 * L_c))
    C55 = (2 * nu / (omega * L_c**2))

    # Raft Points
    L_raft_adim = L_raft / L_c
    L_domain_adim = jnp.floor(L_domain / L_c) - jnp.floor(L_domain/L_c) % 2 + 1 # We make it odd (for simpson integration)
    p = jnp.int32(n * L_domain_adim / L_raft_adim) # total number of points in the domain
    print(f"Total number of points in the domain: {p}")
        
    x = jnp.linspace(-L_domain_adim/2, L_domain_adim/2, p)
    x_contact = abs(x) <= L_raft_adim/2; H = sum(x_contact == True) 
    x_free    = abs(x) >  L_raft_adim/2; x_free = x_free.at[0].set(False); x_free = x_free.at[-1].set(False)
    assert jnp.sum(x_free) + jnp.sum(x_contact) == p-2, f"Number of free and contact points do not match the total number of points {N}." if DEBUG else None
    left_raft_boundary = (p-H)//2; right_raft_boundary = (p+H)//2
    
    if jnp.std(x) < 1e-5:
        grid_x = (x[left_raft_boundary] - x[left_raft_boundary-1]).item(0)
    else:
        grid_x = jnp.round(x, 5)

    # Derivative operators
    d_dx = Diff(axis=0, grid=grid_x, acc=2, shape=(p,))

    # Used for E1, E3 
    # TODO: shape of these might be incorrect
    d_dx_left = Diff(axis=0, grid=grid_x[0:left_raft_boundary], acc=2, shape=(sum(x_free)//2,))
    d_dx_right = Diff(axis=0, grid=grid_x[right_raft_boundary: ], acc=2, shape=(sum(x_free)//2,))

    # Used for E4, E5
    d_dx_raft = Diff(axis=0, grid=grid_x[left_raft_boundary:right_raft_boundary], acc=2, shape=(H,))

    # Building matrix (Ax = b)
    # [E11][0]  [0]   [0]   | [phi]         [0]
    # [E21][0]  [E23] [E24] | [eta]         [0]
    # [E31][E32][0]   [0]   | [zeta]        [0]
    # [E41][E42][E43] [0]   | [theta]       [C42]
    # [E51][E52][0]   [E54] |               [C52]

    # follow equations.md

    E11_left = C11 + (C12 + C14) * d_dx_left**2 + C13
    E11_right = C11 + (C12 + C14) * d_dx_right**2 + C13
    E12 = 0
    E13 = 0
    E14 = 0

    E21 = C21 * N
    E22 = 0
    E23 = C22
    E24 = C23 * x[x_contact]
    print(f"E24: {(C23 * x[x_contact]).shape}")
    
    E31 = C33 * N
    print(f"E32 Left: {(C31 + C32 * d_dx_left**2).shape}")
    print(f"E32 Right: {(C31 + C32 * d_dx_right**2).shape}")
    E32_left = C31 + C32 * d_dx_left**2
    E32_right = C31 + C32 * d_dx_right**2
    E33 = 0
    E34 = 0

    print(f"integral shape: {integral.shape}")  # debugging -> (21, )
    print(f"E41: {(C43 + C45 * d_dx**2).shape}") 

    E41 = integral @ (C43 + C45 * d_dx_raft**2)
    E42 = C44 * integral
    E43 = C41
    E44 = 0

    print(f"integral shape: {integral.shape}") # debugging
    print(f"E5: {(x[x_contact] @ (C53 + C55 * d_dx_raft**2)).shape}")
    E51 = integral @ x[x_contact] @ (C53 + C55 * d_dx_raft**2) 
    E52 = C54 * integral 
    E53 = 0
    E54 = C51

    # building equation blocks
    E1_left = jnp.stack(E11_left, E12, E13, E14, axis = 1)
    E1_right = jnp.stack(E11_right, E12, E13, E14, axis = 1)
    E1 = jnp.stack(E1_left, E1_right, axis = 0) # combine left and right parts

    E2 = jnp.stack(E21, E22, E23, E24, axis = 1)

    E3_left = jnp.stack(E31, E32_left, E33, E34, axis = 1)
    E3_right = jnp.stack(E31, E32_right, E33, E34, axis = 1)
    E3 = jnp.stack(E3_left, E3_right, axis = 0) # combine left and right parts

    E4 = jnp.stack(E41, E42, E43, E44, axis = 1)
    E5 = jnp.stack(E51, E52, E53, E54, axis = 1)

    # Concatenate 
    A = jnp.stack(E1, E2, E3, E4, E5, axis = 0)
    B = jnp.stack(0, 0, 0, C42, C52, axis = 0)

    # Testing E is a square matrix
    [r,c] = A.shape()

    solution = jax.numpy.linalg.solve(A, B)

    # Splitting variables
    # TODO: check splicing for phi specifically
    phi_left = solution[0 : (N - 1)/2]
    phi_right = solution[(N - 1)/2 : N]
    eta = solution[N : (N - n - 1)]
    zeta = solution[N - n]
    theta = solution[N - n + 1]

    return (phi_left, phi_right, eta, zeta, theta, r, c)

if __name__ == "__main__":
    rigidSolver(1000, 10, 1e-6, 9.81, 0.05, 1, 0.072, 0.02, 1, 21)

