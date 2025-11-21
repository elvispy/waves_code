import jax
import numpy as np
from scipy.linalg import qr
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
    Outputs:
    - 
    '''

    ## Derived dimensional Parameters (SI Units)
    L_c = L_raft
    m_c = rho * L_c**2

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
    L = (p - n)//2
    print(f"L points: {L}")
  
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

    # Equation 3 (outside the raft) # TODO: fix this 
    C31 = 1.0
    C32 = 2 * nu / (omega * L_c**2)
    C33 = -1.0j

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

    # Used for E1, E3
    # d_dx_left = Diff(axis=0, grid=grid_x[0:(left_raft_boundary+1)], acc=2, shape=(sum(x_free)//2+1,))

    # why is this only for the left indices here
    d2_dx2_free = (1.0 * Diff(axis=0, grid=grid_x[0:(left_raft_boundary+1)], acc=2, shape=(sum(x_free)//2+1,))**2)[:-1, :]
    # print(f"d_dx_left: {d_dx_left.shape}")
    # print(f"left_raft_boundary: {left_raft_boundary}")
    
    #d_dx_right = Diff(axis=0, grid=grid_x[(right_raft_boundary-1): ], acc=2, shape=(sum(x_free)//2+1,))
    #print(f"d_dx_right: {d_dx_right.shape}")
    print(f"number of free points: {sum(x_free)//2}")

    # Used for E4, E5
    d_dx_raft = Diff(axis=0, grid=grid_x[left_raft_boundary:right_raft_boundary], acc=2, shape=(H,))

    # Building matrix (Ax = b) 
    # [E11_1L] [0]    [0]     [0]     [0]     [0]  [0]         | [phi_left]         [0]
    # [0]     [0]    [E11_2R] [0]     [0]     [0]  [0]         | [phi_center]       [0]
    # [0]     [E21_C][0]     [0]     [0]     [E23][E24]        | [phi_right]        [0]
    # [E31_L][0]    [0]     [E32_L][0]     [0]  [0]            | [eta_left]         [0]
    # [0]     [0]    [E31_R][0]     [E32_R][0]  [0]                             [0]
    # [0]     [E41_C][0]     [0]     [0]     [0]  [E44]       | [eta_right]        [0]
    # [0]     [E51_C][0]     [0]     [0]     [E53]  [0]       | [theta]            [C42]
    #                                                         | [zeta]             [C52]
    # Checking square dimensions: 
    # E1: p-n equations, E2: n equations, E: p-n equations, E4: 1 equation, E5: 1 equation
    # Total equations: 2p - n + 2
    # phi total: p unknowns, eta total: = p-n unknowns, zeta = 1 unknown, theta = 1 unknown
    # Total unknowns: 2p - n + 2

    # Equation 1
    # Bernoulli equation (2L equations)
    E1_Phi  = jnp.vstack([
        jnp.hstack([C13 * jnp.eye(L), jnp.zeros((L, p-L))]),
        jnp.hstack([jnp.zeros((L, p-L)), C13 * jnp.eye(L)])
    ], dtype=jnp.complex64)

    E1_Phi = E1_Phi.at[0:L,   0:(L+1)      ].add(C14 * d2_dx2_free)
    E1_Phi = E1_Phi.at[L:2*L, (p - L - 1):p].add(C14 * d2_dx2_free)
    print("E1_Phi Shape:", E1_Phi.shape)

    E1_Phiz = jnp.vstack([
        jnp.hstack([C11 * jnp.eye(L), jnp.zeros((L, p-L))]),
        jnp.hstack([jnp.zeros((L, p-L)), C11 * jnp.eye(L)])
    ], dtype=jnp.complex64)
    E1_Phiz = E1_Phiz.at[0:L,   0:(L+1)      ].add(C12 * d2_dx2_free)
    E1_Phiz = E1_Phiz.at[L:2*L, (p - L - 1):p].add(C12 * d2_dx2_free)
    print("E1_Phiz Shape:", E1_Phiz.shape)

    # Testing
    print("E1 Phi Shape, First Half:", jnp.hstack([C13 * jnp.eye(L), jnp.zeros((L, p-L))]).shape)
    print("E1 Phi Shape:", E1_Phi.shape)
    
    E1 = jnp.hstack([
        E1_Phi, 
        E1_Phiz, 
        jnp.zeros((2*L, p + 2))
    ], dtype=jnp.complex64)
    
    # Boundary conditions
    k = dispersion_k(omega, g, 100 * L_raft, nu, sigma, rho, k0 = omega**2/g) # Complex wavenumber
    kbar = k * L_c

    # First Index
    E1 = E1.at[0, :].set(0.0)
    E1 = E1.at[0, 0].set(-1.0j * kbar * dx - 1.0) 
    E1 = E1.at[0, 1].set(1.0) 

    # Need to override bc. linear dependence...
    # E1 = E1.at[L + 1:].set(0.0)
    # E1 = E1.at[L + 1:p-L-1].set(-1.0)
    # E1 = E1.at[L + 1:p-L].set(1.0)
    
    # Last Index
    E1 = E1.at[-1, :].set(0.0)
    E1 = E1.at[-1, -2].set(-1.0) 
    E1 = E1.at[-1, -1].set(-1.0j * kbar * dx + 1.0) 

    print("Rank skdjfhsd", jnp.linalg.matrix_rank(E1[1:-1, :]))
    
    # Equation 2 (Kinematic boundary condition inside the raft eta = zeta + x theta)
    E2_Eta   = C21 * jnp.hstack([jnp.zeros((n, L)), jnp.eye(n), jnp.zeros((n, L))], dtype=jnp.complex64)
    E2_Zeta  = C22 * jnp.ones((n, 1))
    E2_Theta = C23 * x[x_contact].reshape((n, 1))

    E2 = jnp.hstack([
        jnp.zeros((n, 2*p)), 
        E2_Eta, 
        E2_Zeta, 
        E2_Theta
    ], dtype=jnp.complex64)

    # Equation 3 (Kinematic boundary condition phi_z = i eta)
    E3_Phiz = jnp.eye(p)
    E3_Eta  = -1.0j * jnp.eye(p)
    E3 = jnp.hstack([
        jnp.zeros((p, p)), 
        E3_Phiz, 
        E3_Eta, 
        jnp.zeros((p, 2))
    ], dtype=jnp.complex64)
    
    # Equation 4
    E4_Phi = jnp.hstack([
        jnp.zeros((1, L)),
        integral @ (C43 + d_dx_raft**2 * C46),
        jnp.zeros((1, L))
    ])
    E4_Theta = (integral @ (C45 * x[x_contact])).reshape(1,1) # theta
    E4_Zeta = C41 + integral @ (jnp.full((n, 1), C44)) # zeta # TODO: check this

    E4 = jnp.hstack([
        E4_Phi,
        jnp.zeros((1, 2*p)), 
        E4_Zeta, 
        E4_Theta
    ], dtype=jnp.complex64)

    #print(f"integral shape: {integral.shape}")
    #print(f"other shape E51C: {(x[x_contact] * (C53 + d_dx_raft**2 * C56)).shape}")
    # Equation 5
    E5_Phi = jnp.hstack([
        jnp.zeros((1, L)),
        integral @ (x[x_contact] * (C53 + d_dx_raft**2 * C56)),
        jnp.zeros((1, L))
    ])
    #E51_C = integral @ (x[x_contact] * (C53 + d_dx_raft**2 * C56)) # phi_center
    E5_Theta = (C51 + integral @ (x[x_contact] * (C55 * x[x_contact]))).reshape(1,1) # theta # TODO: check this @ or * not sure
    E5_Zeta = (integral @ (x[x_contact] * C54)).reshape(1,1) # zeta

    E5 = jnp.hstack([
        E5_Phi,
        jnp.zeros((1, 2*p)), 
        E5_Zeta, 
        E5_Theta
    ], dtype=jnp.complex64)

    # Equation 6 (phiz = Nphi)
    E6_Phi = DtN
    E6_Phiz = -1.0 * jnp.eye(p)

    E6 = jnp.hstack([
        E6_Phi,
        E6_Phiz,
        jnp.zeros((p, p)),
        jnp.zeros((p, 2))
    ], dtype=jnp.complex64)

    # Concatenate 
    A = jnp.vstack([E1, E2, E3, E4, E5, E6], dtype=jnp.complex64)

    b = jnp.vstack([
        jnp.zeros((3*p,1)), 
        C42, 
        C52
    ], dtype=jnp.complex64)

    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")

    print(f"Determinant of A: {jnp.linalg.det(A)}") # alternate: jnp.linalg.slogdet(A) -> as long as ur log is not infinite, okay

    print("Rank of matrix:", jnp.linalg.matrix_rank(A)) # rank need to be equal to the matrix size 
    
    # test = jnp.vstack([E1, E2, E3, E6], dtype=jnp.complex64)
    # print("Rank Test:", jnp.linalg.matrix_rank(test))
    print("Shape of matrix:", A.shape)

    # DEBUGGING: Testing QR Decomposition (using Numpy rather than Jax)
    def find_dependent_rows(A, tol=None):
        """
        Returns:
        independent_rows : list of row indices forming a basis
        dependent_rows   : list of row indices that are linear combinations of the independent ones
        """
        # QR with column pivoting on A^T → pivot rows of A
        Q, R, pivots = qr(A.T, pivoting=True, mode='economic')
        # Determine numerical rank
        if tol is None:
            tol = np.max(A.shape) * np.abs(R).max() * np.finfo(A.dtype).eps
        diag_R = np.abs(np.diag(R))
        rank = np.sum(diag_R > tol)
        independent_rows = sorted(pivots[:rank])
        dependent_rows   = sorted(pivots[rank:])

        return independent_rows, dependent_rows
    ind_rows, dep_rows = find_dependent_rows(np.asarray(A)) 
    print("Dependent row indices:", dep_rows) 

    ind_rows, dep_rows_dx2 = find_dependent_rows(np.asarray(d2_dx2_free)) 
    print("Dependent row indices, dx2:", dep_rows_dx2) 

    if jnp.linalg.matrix_rank(A) < min(A.shape):
        print("Matrix has linearly dependent rows or columns.")
    else: 
        print("No linear dependence.")
   
    solution = jax.numpy.linalg.solve(A, b)

    # Splitting variables 
    phi = solution[0 : p]
    eta = solution[(-p-2) : -2]
    zeta = solution[-2]
    theta = solution[-1]

    # returning A now for testing...
    return [phi, eta, zeta, theta, A]

if __name__ == "__main__":
    A = rigidSolver(1000, 10, 1e-6, 9.81, 0.05, 1, 0.072, 0.02, 1, 21)
