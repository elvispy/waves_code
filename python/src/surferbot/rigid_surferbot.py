import jax
import jax.numpy as jnp
from surferbot.DtN import DtN_generator
from surferbot.myDiff import Diff
from surferbot.integration import simpson_weights
from surferbot.utils import dispersion_k
from surferbot.constants import DEBUG


def rigidSolver(rho, omega, nu, g, L_raft, L_domain, sigma, x_A, F_A, n):
    '''
    Inputs: 
    - rho: density of fluid (kg/m^2)
    - omega: angular frequency (rad/s)
    - g: gravity (m/s^2)
    - L_raft: length of raft (m)
    - L_domain: length of domain (m)
    - motor_inertia: inertia of the motor (kg*m^2)
    - nu: kinematic viscosity (m^2/s)
    - sigma: surface tension (N)
    - n: number of points in raft
    - p: total number of points in domain (x-direction) # TODO: probably change this to not get mixed up with the other p in type-up
    Outputs:
    - 
    '''

    ## Derived dimensional Parameters (SI Units)
    L_c = L_raft
    m_c = rho * L_c**2
    t_c = 1/omega

    # Raft Points
    L_raft_adim = L_raft / L_c
    L_domain_adim = jnp.floor(L_domain / L_c) - jnp.floor(L_domain/L_c) % 2 + 1 # We make it odd (for simpson integration)
    p = jnp.int32(n * L_domain_adim / L_raft_adim) # total number of points in the domain
    print(f"Total number of points in the domain: {p}")

    # Equation setup
    DtN = DtN_generator(p)
    dx = 1/n
    N = DtN / (L_raft / n) # operator
    print(f"N shape: {N.shape}")

    # Splitting N 
    N11 = N[0 : (p - n)//2, 0 : (p - n)//2]
    N12 = N[0 : (p - n)//2, (p - n)//2 : (p - n)//2 + n]
    N13 = N[0 : (p - n)//2, (p - n)//2 + n : p]

    N21 = N[(p - n)//2 : (p - n)//2 + n, 0 : (p - n)//2]
    N22 = N[(p - n)//2 : (p - n)//2 + n, (p - n)//2 : (p - n)//2 + n]
    N23 = N[(p - n)//2 : (p - n)//2 + n, (p - n)//2 + n : p]
    
    N31 = N[(p - n)//2 + n : p, 0 : (p - n)//2]
    N32 = N[(p - n)//2 + n : p, (p - n)//2 : (p - n)//2 + n]
    N33 = N[(p - n)//2 + n : p, (p - n)//2 + n : p]
            
    integral = simpson_weights(n, dx).reshape(1, n)  # Ensure integral is a row vector

    # Equation 1 (Bernoulli equation)
    C11 = 1.0
    C12 = -(sigma / (rho * L_c**3 * omega**2))/(g / (L_c * omega**2))
    C13 = -(omega**2 * L_c / g)      
    C14 = -(4.0j * nu / (omega * L_c**2)) / (g / (L_c * omega**2))

    # Kinematic boundary conditions
    # Equation 2 (inside the raft) 
    C21 = 1.0   # N operator accounted for in equation building
    C22 = -1.0j
    C23 = -1.0j 

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
    C45 = 0.5 * g / (omega**2 * L_c)
    C46 = 2 * nu / (omega * L_c**2)

    # Equation 5 
    C51 = -(1/12) * m_c
    C52 = -(x_A / L_c) * (F_A / (rho * L_c**3 * omega**2)) # constant term
    C53 = 1.0j
    C54 = (g / (omega**2 * L_c))
    C55 = 0.5 * (g / (omega**2 * L_c))
    C56 = (2 * nu / (omega * L_c**2))

    # Grid setup
    x = jnp.linspace(-L_domain_adim/2, L_domain_adim/2, p)
    x_contact = abs(x) <= L_raft_adim/2; H = sum(x_contact == True) 
    x_free    = abs(x) >  L_raft_adim/2
    left_raft_boundary = (p-H)//2; right_raft_boundary = left_raft_boundary + H
  
    if jnp.std(x) < 1e-5:
        grid_x = (x[left_raft_boundary] - x[left_raft_boundary-1]).item(0)
    else:
        grid_x = jnp.round(x, 5)

    # Derivative operators
    d_dx = Diff(axis=0, grid=grid_x, acc=2, shape=(p,))

    # Used for E1, E3 
    d_dx_left = Diff(axis=0, grid=grid_x[0:left_raft_boundary], acc=2, shape=(sum(x_free)//2,))
    d_dx_right = Diff(axis=0, grid=grid_x[right_raft_boundary: ], acc=2, shape=(sum(x_free)//2,))

    # Used for E4, E5
    d_dx_raft = Diff(axis=0, grid=grid_x[left_raft_boundary:right_raft_boundary], acc=2, shape=(H,))

    # Building matrix (Ax = b)
    # [E11_L] [0]    [0]     [0]     [0]     [0]  [0]         | [phi_left]         [0]
    # [0]     [0]    [E11_R] [0]     [0]     [0]  [0]         | [phi_center]       [0]
    # [0]     [E21_C][0]     [0]     [0]     [E23][E24]       | [phi_right]        [0]
    # [E31_1L][0]    [0]     [E32_1L][0]     [0]  [0]         | [eta_left]         [0]
    # [0]     [0]    [E31_2R][0]     [E32_2R][0]  [0]         | [eta_center]       [0]
    # [0]     [E41_C][0]     [0]     [0]     [0]  [E44]       | [eta_right]        [0]
    # [0]     [E51_C][0]     [0]     [0]     [E53]  [0]       | [theta]            [C42]
    #                                                         | [zeta]             [C52]
    # Checking square dimensions: 
    # E1: p-n equations, E2: n equations, E: p-n equations, E4: 1 equation, E5: 1 equation
    # Total equations: 2p - n + 2
    # phi total: p unknowns, eta total: = p-n unknowns, zeta = 1 unknown, theta = 1 unknown
    # Total unknowns: 2p - n + 2

    # Equation 1
    E11_L = C11 * N11 + d_dx_left**2 * (C12 * N11 + C14) + C13 # phi_left
    print(f"E11_1L: {E11_L.shape}")
    E11_R = C11 * N11 + d_dx_right**2 * (C12 * N11 + C14) + C13 # phi_right

    # Equation 2
    E21_C = C21 * N22 # phi_center
    print(f"E21_C: {E21_C.shape}")
    E23 = (C23 * x[x_contact]).reshape((n, 1)) # theta
    E24 = jnp.full((n, 1), C22) # zeta

    # Equation 3
    E31_L = C31 * N31 # phi_left
    E31_R = C33 * N33 # phi_left, phi_right
    print(f"E31_L: {E31_L.shape}") 
    E32_L = C31 + d_dx_left**2 * C32 # eta_left
    E32_R = C31 + d_dx_right**2 * C32 # eta_right

    # Equation 4
    E41_C = integral @ (C43 + d_dx_raft**2 * C46) # phi_center
    print(f"integral shape: {integral.shape}")
    print(f"other shape E41C: {(C43 + d_dx_raft**2 * C46).shape}")

    E43 = (integral @ (C45 * x[x_contact])).reshape(1,1) # theta
    E44 = C41 + integral @ (jnp.full((n, 1), C44)) # zeta # TODO: check this

    print(f"integral shape: {integral.shape}")
    print(f"other shape E51C: {(x[x_contact] * (C53 + d_dx_raft**2 * C56)).shape}")
    # Equation 5
    E51_C = integral @ (x[x_contact] * (C53 + d_dx_raft**2 * C56)) # phi_center
    E53 = (C51 + integral @ (x[x_contact] * (C55 * x[x_contact]))).reshape(1,1) # theta # TODO: check this @ or * not sure
    E54 = (integral @ (x[x_contact] * C54)).reshape(1,1) # zeta

    # Stacking equations
    # Zero matrices
    O_NP = jnp.zeros((n, (p - n)//2))          # zero matrix of size (n) x (p-n)
    O_PP = jnp.zeros(((p - n)//2, (p - n)//2)) # zero matrix of size (p-n)/2 x (p-n)/2
    O_P1 = jnp.zeros(((p - n)//2, 1))          # zero matrix of size (p-n)/2 x 1
    O_N1 = jnp.zeros((n, 1))                   # zero matrix of size (n) x 1
    O_1P = jnp.zeros((1, (p - n)//2))          # zero matrix of size 1 x (p-n)/2

    # E1
    print(f"E1 shape check: {E11_L.shape}, {E11_R.shape}") 
    E1_L = jnp.hstack([E11_L, N12, N13, O_PP, O_PP, O_P1, O_P1])
    E1_R = jnp.hstack([N11, N12, E11_R, O_PP, O_PP, O_P1, O_P1])

    # E2 
    E2 = jnp.hstack([N21, E21_C, N23, O_NP, O_NP, E23, E24])

    # E3
    E3_L = jnp.hstack([E31_L, N32, N33, E32_L, O_PP, O_P1, O_P1])
    E3_R = jnp.hstack([N31, N32, E31_R, O_PP, E32_R, O_P1, O_P1])

    # Boundary conditions
    k = dispersion_k(omega, g, 0.0, nu, sigma, rho) # Complex wavenumber (0.0 should be infinity / very large)

    # need to make everything nondimensional
    E3_L = E3_L.at[0].set(jnp.zeros(2 * p - n + 2)) # Zeroing last row
    E3_R = E3_R.at[-1].set(jnp.zeros(2 * p - n + 2)) # Zeroing last row

    E3_L = E3_L.at[0, 0].set(-1.0j*k - (1.0 / dx)) # -1.0j * k * dx - 1.0
    E3_L = E3_L.at[0, 1].set(1.0 / dx) # 1.0
    E3_R = E3_R.at[-1, -2].set(1.0j*k - (1.0 / dx)) # -1.0
    E3_R = E3_R.at[-1, -1].set(1.0 / dx) # -1.0j * k * dx + 1.0

    # E4, E5
    E4 = jnp.hstack([O_1P, E41_C, O_1P, O_1P, O_1P, E43, E44])
    E5 = jnp.hstack([O_1P, E51_C, O_1P, O_1P, O_1P, E53, E54])

    print(f"E1_L shape: {E1_L.shape}")
    print(f"E1_R shape: {E1_R.shape}")
    print(f"E2 shape: {E2.shape}")
    print(f"E3_L shape: {E3_L.shape}")
    print(f"E3_R shape: {E3_R.shape}")
    print(f"E4 shape: {E4.shape}")
    print(f"E5 shape: {E5.shape}")  

    # Concatenate 
    A = jnp.vstack([E1_L, E1_R, E2, E3_L, E3_R, E4, E5])
    b = jnp.vstack([O_P1, O_N1, O_P1, O_P1, O_P1, C42, C52])

    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")

    print(f"Determinant of A: {jnp.linalg.det(A)}")
    print("Any zero rows:", jnp.any(jnp.all(A == 0, axis=1)))
    print("Any zero columns:", jnp.any(jnp.all(A == 0, axis=0)))

    print("Rank of matrix:", jnp.linalg.matrix_rank(A))
    print("Shape of matrix:", A.shape)
    if jnp.linalg.matrix_rank(A) < min(A.shape):
        print("Matrix has linearly dependent rows or columns.")

    solution = jax.numpy.linalg.solve(A, b)

    # Splitting variables
    phi = solution[0 : p]
    eta = solution[p : p - n]
    zeta = solution[p - n]
    theta = solution[p - n + 1]

    return (phi, eta, zeta, theta)

if __name__ == "__main__":
    A = rigidSolver(1000, 10, 1e-6, 9.81, 0.05, 1, 0.072, 0.02, 1, 21)
    print(A[0])
