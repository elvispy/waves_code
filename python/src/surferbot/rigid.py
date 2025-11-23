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
    L_c = L_raft                   # characteristic length taken as raft length
    m_c = rho * L_c**2             # characteristic mass scale

    # Raft / domain discretization in non-dimensional units
    L_raft_adim = L_raft / L_c
    # Make domain length an odd integer multiple of L_c so Simpson integration weights are valid
    L_domain_adim = jnp.floor(L_domain / L_c) - jnp.floor(L_domain / L_c) % 2 + 1
    p = jnp.int32(n * L_domain_adim / L_raft_adim)  # total number of grid points in the DtN domain
    print(f"Total number of points in the domain: {p}")

    # Equation setup
    DtN = DtN_generator(p)         # DtN operator mapping phi -> phi_z on free surface
    dx = 1 / n                     # non-dimensional spacing on the raft
    N = DtN / (L_raft / n)         # rescaled DtN operator consistent with dimensionalization
    print(f"N shape: {N.shape}")

    # Splitting N / domain into left-free, raft, right-free parts
    L = (p - n) // 2               # number of free-surface points on each side of raft
    print(f"L points: {L}")
  
    # Simpson integration weights over raft (row vector)
    integral = simpson_weights(n, dx).reshape(1, n)

    # Bernoulli equation coefficients (relating phi, phi_z, eta and viscous terms)
    C11 = 1.0
    C12 = -(sigma / (rho * L_c**3 * omega**2)) / (g / (L_c * omega**2))
    C13 = -(omega**2 * L_c / g)      
    C14 = -(4.0j * nu / (omega * L_c**2)) / (g / (L_c * omega**2))

    # Kinematic boundary condition inside the raft (eta = zeta + x * theta)
    C21 = 1.0   # coefficient for eta
    C22 = -1.0j # coefficient for zeta
    C23 = -1.0j # coefficient for theta

    # Kinematic relation for free surface (not directly used in current matrix layout)
    C31 = 1.0
    C32 = 2 * nu / (omega * L_c**2)
    C33 = -1.0j

    # Rigid-body Newton equations (horizontal and rotational balance)
    # Equation 4 (horizontal force balance)
    C41 = -m_c
    C42 = -F_A / (rho * L_c**3 * omega**2)  # constant forcing term in RHS
    C43 = 1.0j
    C44 = g / (omega**2 * L_c) 
    C45 = 0.5 * g / (omega**2 * L_c)
    C46 = 2 * nu / (omega * L_c**2)

    # Equation 5 (moment balance about raft center)
    C51 = -(1/12) * m_c
    C52 = -(x_A / L_c) * (F_A / (rho * L_c**3 * omega**2))  # moment from forcing
    C53 = 1.0j
    C54 = (g / (omega**2 * L_c))
    C55 = 0.5 * (g / (omega**2 * L_c))
    C56 = (2 * nu / (omega * L_c**2))

    # Grid setup in non-dimensional coordinates
    x = jnp.linspace(-L_domain_adim/2, L_domain_adim/2, p)
    x_contact = abs(x) <= L_raft_adim/2  # indices where raft is in contact with fluid
    H = sum(x_contact == True)          # number of raft points (should equal n)
    x_free = abs(x) > L_raft_adim/2     # free-surface points (outside raft)
    left_raft_boundary = (p - H) // 2
    right_raft_boundary = left_raft_boundary + H
  
    # Determine grid spacing / grid array used for finite-difference operators
    if jnp.std(x) < 1e-5:
        # If all x are almost identical (degenerate case), recover spacing directly
        grid_x = (x[left_raft_boundary] - x[left_raft_boundary-1]).item(0)
    else:
        # Use rounded x as non-uniform grid support for Diff operators
        grid_x = jnp.round(x, 5)

    # Second derivative on free surface (used in Bernoulli / viscous terms on left/right sides)
    d2_dx2_free = (1.0 * Diff(axis=0,
                              grid=grid_x[0:(left_raft_boundary+1)],
                              acc=2,
                              shape=(sum(x_free)//2+1,))**2)[:-1, :]
    print(f"number of free points: {sum(x_free)//2}")

    # First/second derivatives on raft (used in integrated Newton equations)
    d_dx_raft = Diff(axis=0,
                     grid=grid_x[left_raft_boundary:right_raft_boundary],
                     acc=2,
                     shape=(H,))


    # Building matrix (A x = b)
    #
    # We conceptually split the p grid points as:
    #   left free surface  : L points
    #   raft (contact)     : n points
    #   right free surface : L points
    # so that, for any field q,
    #   q = [q_L (L); q_C (n); q_R (L)]
    #
    # Unknown vector (conceptual block form):
    #   x = [
    #        phi_L    (L)
    #        phi_C    (n)
    #        phi_R    (L)
    #        phi_z_L  (L)
    #        phi_z_C  (n)
    #        phi_z_R  (L)
    #        eta_L    (L)
    #        eta_C    (n)
    #        eta_R    (L)
    #        zeta     (1)
    #        theta    (1)
    #       ]
    #
    # Row blocks (in stacking order):
    #   E1: Bernoulli equation on free-surface regions (left/right only)
    #   E2: Kinematic condition on raft (eta_C = zeta + x * theta)
    #   E3: Global kinematic condition (phi_z = i * eta at all p points)
    #   E4: Integrated horizontal force balance
    #   E5: Integrated moment balance
    #   E6: DtN relation (phi_z = N phi at all p points)
    #
    # Block structure (schematic):
    #
    # [E11_L   0       E11_R   E12_L   0       E12_R   0        0        0        0     0]   | E1 (Bernoulli on free surface)
    # [ 0      0        0       0      0        0      0      E21_C      0       E22   E23]   | E2 (eta_C - zeta - x*theta = 0)
    # [ 0      0        0      E31_L  E31_C   E31_R  E32_L   E32_C    E32_R      0     0]   | E3 (phi_z - i*eta = 0 on all p)
    # [ 0     E41_C     0       0      0        0      0        0        0      E43   E44]   | E4 (horizontal force balance)
    # [ 0     E51_C     0       0      0        0      0        0        0      E53   E54]   | E5 (moment balance)
    # [E61_L E61_C   E61_R   E62_L  E62_C   E62_R    0        0        0        0     0]   | E6 (phi_z - N*phi = 0)
    #
    # Dimensions / counts:
    #   E1: 2L equations   (Bernoulli on p - n = 2L free-surface points)
    #   E2: n  equations   (raft kinematic condition)
    #   E3: p  equations   (global kinematic relation)
    #   E4: 1  equation    (horizontal force balance)
    #   E5: 1  equation    (moment balance)
    #   E6: p  equations   (DtN relation)
    #   ----------------------------------------------
    #   Total equations: 2L + n + p + 1 + 1 + p = 3p + 2
    #
    #   phi   : p unknowns (L + n + L)
    #   phi_z : p unknowns (L + n + L)
    #   eta   : p unknowns (L + n + L)
    #   zeta  : 1 unknown
    #   theta : 1 unknown
    #   ----------------------------------------------
    #   Total unknowns: 3p + 2


    # Building matrix (Ax = b)
    #
    # Unknown vector ordering:
    #   [phi (p); phi_z (p); eta (p); zeta; theta]
    #
    # Blocks E1–E6 correspond to:
    #   E1: Bernoulli equation on free surface (p - n = 2L equations)
    #   E2: Kinematic condition on raft (n equations, eta = zeta + x * theta)
    #   E3: Global kinematic condition (phi_z = i * eta) on entire domain (p equations)
    #   E4: Integrated horizontal force balance (1 equation)
    #   E5: Integrated moment balance (1 equation)
    #   E6: DtN relation (phi_z = N * phi) on entire domain (p equations)
    #
    # Total equations: (p - n) + n + p + 1 + 1 + p = 3p + 2
    # Total unknowns:  phi (p) + phi_z (p) + eta (p) + zeta (1) + theta (1) = 3p + 2

    # Equation 1: Bernoulli equation on free-surface regions (left and right)
    E1_Phi  = jnp.vstack([
        jnp.hstack([C13 * jnp.eye(L), jnp.zeros((L, p-L))]),
        jnp.hstack([jnp.zeros((L, p-L)), C13 * jnp.eye(L)])
    ], dtype=jnp.complex64)

    # Add viscous term (second derivative) on each side of the free surface
    E1_Phi = E1_Phi.at[0:L,   0:(L+1)      ].add(C14 * d2_dx2_free)
    E1_Phi = E1_Phi.at[L:2*L, (p - L - 1):p].add(C14 * d2_dx2_free)
    print("E1_Phi Shape:", E1_Phi.shape)

    # Coefficients multiplying phi_z in Bernoulli equation
    E1_Phiz = jnp.vstack([
        jnp.hstack([C11 * jnp.eye(L), jnp.zeros((L, p-L))]),
        jnp.hstack([jnp.zeros((L, p-L)), C11 * jnp.eye(L)])
    ], dtype=jnp.complex64)
    # Add surface tension term via second derivative of phi_z
    E1_Phiz = E1_Phiz.at[0:L,   0:(L+1)      ].add(C12 * d2_dx2_free)
    E1_Phiz = E1_Phiz.at[L:2*L, (p - L - 1):p].add(C12 * d2_dx2_free)
    print("E1_Phiz Shape:", E1_Phiz.shape)

    # Testing shapes
    print("E1 Phi Shape, First Half:", jnp.hstack([C13 * jnp.eye(L), jnp.zeros((L, p-L))]).shape)
    print("E1 Phi Shape:", E1_Phi.shape)
    
    # Assemble E1 block over all unknowns: [phi, phi_z, eta, zeta, theta]
    E1 = jnp.hstack([
        E1_Phi, 
        E1_Phiz, 
        jnp.zeros((2*L, p + 2))
    ], dtype=jnp.complex64)
    
    # Radiation / boundary conditions via complex wavenumber
    k = dispersion_k(omega, g, 100 * L_raft, nu, sigma, rho, k0=omega**2/g)  # Complex wavenumber
    kbar = k * L_c

    # Upstream radiation condition at left boundary
    E1 = E1.at[0, :].set(0.0)
    E1 = E1.at[0, 0].set(-1.0j * kbar * dx - 1.0) 
    E1 = E1.at[0, 1].set(1.0) 

    # Downstream radiation condition at right boundary
    E1 = E1.at[-1, :].set(0.0)
    E1 = E1.at[-1, -2].set(-1.0) 
    E1 = E1.at[-1, -1].set(-1.0j * kbar * dx + 1.0) 

    print("Rank skdjfhsd", jnp.linalg.matrix_rank(E1[1:-1, :]))
    
    # Equation 2: kinematic condition on raft (eta = zeta + x * theta) enforced on contact points
    E2_Eta   = C21 * jnp.hstack([jnp.zeros((n, L)), jnp.eye(n), jnp.zeros((n, L))], dtype=jnp.complex64)
    E2_Zeta  = C22 * jnp.ones((n, 1))
    E2_Theta = C23 * x[x_contact].reshape((n, 1))

    E2 = jnp.hstack([
        jnp.zeros((n, 2*p)), 
        E2_Eta, 
        E2_Zeta, 
        E2_Theta
    ], dtype=jnp.complex64)

    # Equation 3: global kinematic boundary condition phi_z = i * eta
    E3_Phiz = jnp.eye(p)
    E3_Eta  = -1.0j * jnp.eye(p)
    E3 = jnp.hstack([
        jnp.zeros((p, p)), 
        E3_Phiz, 
        E3_Eta, 
        jnp.zeros((p, 2))
    ], dtype=jnp.complex64)
    
    # Equation 4: integrated horizontal force balance (translation)
    E4_Phi = jnp.hstack([
        jnp.zeros((1, L)),
        integral @ (C43 + d_dx_raft**2 * C46),
        jnp.zeros((1, L))
    ])
    E4_Theta = (integral @ (C45 * x[x_contact])).reshape(1, 1)  # contribution from pitch
    # Heave/zeta term including weight through C44
    E4_Zeta = C41 + integral @ (jnp.full((n, 1), C44))

    E4 = jnp.hstack([
        E4_Phi,
        jnp.zeros((1, 2*p)), 
        E4_Zeta, 
        E4_Theta
    ], dtype=jnp.complex64)

    # Equation 5: integrated moment balance about raft center (rotation)
    E5_Phi = jnp.hstack([
        jnp.zeros((1, L)),
        integral @ (x[x_contact] * (C53 + d_dx_raft**2 * C56)),
        jnp.zeros((1, L))
    ])
    E5_Theta = (C51 + integral @ (x[x_contact] * (C55 * x[x_contact]))).reshape(1, 1)
    E5_Zeta = (integral @ (x[x_contact] * C54)).reshape(1, 1)

    E5 = jnp.hstack([
        E5_Phi,
        jnp.zeros((1, 2*p)), 
        E5_Zeta, 
        E5_Theta
    ], dtype=jnp.complex64)

    # Equation 6: DtN relation enforcing phi_z = N * phi at all p points
    E6_Phi = DtN
    E6_Phiz = -1.0 * jnp.eye(p)

    E6 = jnp.hstack([
        E6_Phi,
        E6_Phiz,
        jnp.zeros((p, p)),
        jnp.zeros((p, 2))
    ], dtype=jnp.complex64)

    # Concatenate all equation blocks into full system A x = b
    A = jnp.vstack([E1, E2, E3, E4, E5, E6], dtype=jnp.complex64)

    # Right-hand side: forcing only appears in rigid-body equations (C42, C52)
    b = jnp.vstack([
        jnp.zeros((3*p, 1)), 
        C42, 
        C52
    ], dtype=jnp.complex64)

    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")

    # Diagnostics: determinant and rank of system matrix
    print(f"Determinant of A: {jnp.linalg.det(A)}")
    print("Rank of matrix:", jnp.linalg.matrix_rank(A)) 
    print("Shape of matrix:", A.shape)

    # DEBUGGING: Testing QR Decomposition (using Numpy rather than Jax)
    def find_dependent_rows(A, tol=None):
        """
        Identify linearly independent and dependent rows via QR decomposition.

        Returns:
        independent_rows : list of row indices that form a basis
        dependent_rows   : list of row indices that are linear combinations of the independent ones
        """
        # QR with column pivoting on A^T → pivots correspond to row indices of A
        Q, R, pivots = qr(A.T, pivoting=True, mode='economic')
        # Determine numerical rank
        if tol is None:
            tol = np.max(A.shape) * np.abs(R).max() * np.finfo(A.dtype).eps
        diag_R = np.abs(np.diag(R))
        rank = np.sum(diag_R > tol)
        independent_rows = sorted(pivots[:rank])
        dependent_rows   = sorted(pivots[rank:])

        return independent_rows, dependent_rows

    # Check for dependence in the full system matrix
    ind_rows, dep_rows = find_dependent_rows(np.asarray(A)) 
    print("Dependent row indices:", dep_rows) 

    # Check for dependence specifically in second-derivative free-surface operator
    ind_rows, dep_rows_dx2 = find_dependent_rows(np.asarray(d2_dx2_free)) 
    print("Dependent row indices, dx2:", dep_rows_dx2) 

    if jnp.linalg.matrix_rank(A) < min(A.shape):
        print("Matrix has linearly dependent rows or columns.")
    else: 
        print("No linear dependence.")
   
    # Solve linear system for all unknowns
    solution = jax.numpy.linalg.solve(A, b)

    # Extract variables from solution vector
    phi = solution[0 : p]
    eta = solution[(-p-2) : -2]
    zeta = solution[-2]
    theta = solution[-1]

    # Return fields and system matrix for downstream analysis
    return [phi, eta, zeta, theta, A]


if __name__ == "__main__":
    A = rigidSolver(1000, 10, 1e-6, 9.81, 0.05, 1, 0.072, 0.02, 1, 21)

