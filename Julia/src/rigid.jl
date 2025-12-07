module Rigid

using LinearAlgebra
using SparseArrays
using FiniteDifferences
using Printf

# Include our local modules
# In a real package, these would be `using ..DtN`, etc.
# For this script, we assume they are available in the load path or included.
# We will assume this file is part of the `Surferbot` package or included after them.

include("DtN.jl")
include("integration.jl")
include("utils.jl")

using .DtN
using .Integration
using .Utils

export rigidSolver


"""
    Diff(grid; acc=2, derivative=1)

Construct a differentiation matrix using FiniteDifferences.jl coefficients.
"""
function Diff(grid::AbstractVector{Float64}; acc::Int=2, derivative::Int=1)
    N = length(grid)
    dxs = diff(grid)
    
    if !all(x -> isapprox(x, dxs[1], atol=1e-6), dxs)
        @warn "Grid appears non-uniform. Using average spacing."
    end
    
    dx = sum(dxs) / length(dxs)
    
    # Get coefficients from FiniteDifferences.jl
    # central_fdm(order, deriv)
    # forward_fdm(points, deriv) -> points = order + 1 for 1st deriv?
    # For acc=2 (2nd order):
    # Central: central_fdm(2, 1) -> [-0.5, 0.5] (2 points)
    # Forward: forward_fdm(3, 1) -> [-1.5, 2.0, -0.5] (3 points)
    
    # Interior method
    m_central = central_fdm(acc, derivative)
    grid_central = m_central.grid
    coefs_central = m_central.coefs
    
    # Boundary methods (Forward/Backward)
    # We need acc order. Forward/Backward usually need acc+1 points for acc order?
    # forward_fdm(p, q): p is number of points?
    # Let's assume p = acc + derivative? Or just acc + 1?
    # For deriv=1, acc=2 -> 3 points.
    # For deriv=2, acc=2 -> 4 points? (-1, 4, -5, 2) / h^2?
    # Let's try to request enough points.
    
    # Heuristic: use acc + derivative points?
    # forward_fdm(3, 1) -> 2nd order.
    # forward_fdm(4, 2) -> 2nd order?
    
    # Let's just use a safe number of points or rely on standard_central_coefs logic if exposed.
    # Since we can't easily map "acc" to "points" genericallly without testing,
    # we will use a lookup or just `acc + derivative`?
    
    pts_bound = acc + derivative # e.g. 2+1=3 for 1st deriv, 2+2=4 for 2nd deriv
    m_forward = forward_fdm(pts_bound, derivative)
    m_backward = backward_fdm(pts_bound, derivative)
    
    D = spzeros(Float64, N, N)
    
    # Helper to fill row
    function fill_row!(row_idx, method)
        g = method.grid # relative indices
        c = method.coefs
        for (k, offset) in enumerate(g)
            col_idx = row_idx + offset
            if 1 <= col_idx <= N
                D[row_idx, col_idx] = c[k] / (dx^derivative)
            end
        end
    end
    
    # Fill matrix
    half_width = div(length(grid_central)-1, 2) # e.g. 2 points -> 0? No.
    # central_fdm(2, 1) -> grid [-1, 1]. half_width = 1?
    # If grid is [-1, 1], we can apply it at i if i-1 >= 1 and i+1 <= N.
    # i.e. i >= 2 and i <= N-1.
    
    min_offset = minimum(grid_central)
    max_offset = maximum(grid_central)
    
    for i in 1:N
        if i + min_offset >= 1 && i + max_offset <= N
            # Interior
            fill_row!(i, m_central)
        elseif i + min_offset < 1
            # Left boundary
            fill_row!(i, m_forward)
        else
            # Right boundary
            fill_row!(i, m_backward)
        end
    end
    
    return D
end

function rigidSolver(rho, omega, nu, g, L_raft, L_domain, sigma, x_A, F_A, n)
    # Derived dimensional Parameters (SI Units)
    L_c = L_raft                   # characteristic length taken as raft length
    m_c = rho * L_c^2             # characteristic mass scale

    # Raft / domain discretization in non-dimensional units
    L_raft_adim = L_raft / L_c
    # Make domain length an odd integer multiple of L_c so Simpson integration weights are valid
    L_domain_adim = floor(L_domain / L_c) - (floor(L_domain / L_c) % 2) + 1
    p = Int(n * L_domain_adim / L_raft_adim)  # total number of grid points in the DtN domain
    println("Total number of points in the domain: $p")

    # Equation setup
    dtn = DtN_generator(p)         # DtN operator mapping phi -> phi_z on free surface
    dx = 1 / n                     # non-dimensional spacing on the raft
    dtn = dtn / (L_raft / n)         # rescaled DtN operator consistent with dimensionalization
    println("N shape: $(size(dtn))")

    # Splitting N / domain into left-free, raft, right-free parts
    L = div(p - n, 2)               # number of free-surface points on each side of raft
    println("L points: $L")
  
    # Simpson integration weights over raft (row vector)
    integral = reshape(simpson_weights(n, Float64(dx)), 1, n)

    # Bernoulli equation coefficients (relating phi, phi_z, eta and viscous terms)
    C11 = 1.0
    C12 = -(sigma / (rho * L_c^3 * omega^2)) / (g / (L_c * omega^2))
    C13 = -(omega^2 * L_c / g)      
    C14 = -(4.0im * nu / (omega * L_c^2)) / (g / (L_c * omega^2))

    # Kinematic boundary condition inside the raft (eta = zeta + x * theta)
    C21 = 1.0   # coefficient for eta
    C22 = -1.0im # coefficient for zeta
    C23 = -1.0im # coefficient for theta

    # Kinematic relation for free surface (not directly used in current matrix layout)
    C31 = 1.0
    C32 = 2 * nu / (omega * L_c^2)
    C33 = -1.0im

    # Rigid-body Newton equations (horizontal and rotational balance)
    # Equation 4 (horizontal force balance)
    C41 = -m_c
    C42 = -F_A / (rho * L_c^3 * omega^2)  # constant forcing term in RHS
    C43 = 1.0im
    C44 = g / (omega^2 * L_c) 
    C45 = 0.5 * g / (omega^2 * L_c)
    C46 = 2 * nu / (omega * L_c^2)

    # Equation 5 (moment balance about raft center)
    C51 = -(1/12) * m_c
    C52 = -(x_A / L_c) * (F_A / (rho * L_c^3 * omega^2))  # moment from forcing
    C53 = 1.0im
    C54 = (g / (omega^2 * L_c))
    C55 = 0.5 * (g / (omega^2 * L_c))
    C56 = (2 * nu / (omega * L_c^2))

    # Grid setup in non-dimensional coordinates
    x = range(-L_domain_adim/2, stop=L_domain_adim/2, length=p)
    x_contact = abs.(x) .<= L_raft_adim/2  # indices where raft is in contact with fluid
    H = sum(x_contact)          # number of raft points (should equal n)
    x_free = abs.(x) .> L_raft_adim/2     # free-surface points (outside raft)
    left_raft_boundary = div(p - H, 2) + 1 # 1-based index
    right_raft_boundary = left_raft_boundary + H - 1
  
    # Determine grid spacing / grid array used for finite-difference operators
    grid_x = x[left_raft_boundary] - x[left_raft_boundary-1]
    
    # Second derivative on free surface (used in Bernoulli / viscous terms on left/right sides)
    grid_free_left = collect(x[1:left_raft_boundary]) # Include boundary point
    
    D1_free = Diff(grid_free_left, acc=2, derivative=1)
    D2_free_sq = D1_free^2
    
    d2_dx2_free = D2_free_sq[1:end-1, :]
    
    println("number of free points: $(div(sum(x_free), 2))")

    # First/second derivatives on raft (used in integrated Newton equations)
    grid_raft = collect(x[left_raft_boundary:right_raft_boundary])
    d_dx_raft = Diff(grid_raft, acc=2, derivative=1)

    # Building matrix (A x = b)
    # This section constructs the linear system A*x = b, where x is the vector of unknowns.
    # The unknowns are ordered as follows:
    # x = [phi_free_left; phi_raft; phi_free_right;      (velocity potential on the entire domain)
    #      phi_z_free_left; phi_z_raft; phi_z_free_right; (vertical derivative of velocity potential)
    #      eta_raft;                                      (free surface elevation under the raft)
    #      zeta;                                          (vertical displacement of the raft)
    #      theta]                                         (angular displacement of the raft)
    #
    # The matrix A is assembled from several blocks, each representing a physical equation:
    #
    # E1: Bernoulli Equation on the free surface (left and right of the raft).
    #     This equation relates the velocity potential (phi), its vertical derivative (phi_z),
    #     and their spatial derivatives, incorporating terms for gravity, surface tension,
    #     and viscosity. It applies to the fluid domain outside the raft.
    #
    # E2: Kinematic Boundary Condition on the raft.
    #     This equation relates the free surface elevation under the raft (eta_raft) to the
    #     rigid body motions of the raft (vertical displacement zeta and angular displacement theta).
    #     Specifically, eta_raft = zeta + x * theta.
    #
    # E3: Dirichlet-to-Neumann (DtN) Operator on the free surface.
    #     This operator provides a relationship between the velocity potential (phi) and its
    #     normal derivative (phi_z) on the free surface, effectively acting as a boundary condition
    #     that accounts for wave propagation to infinity. It applies to the fluid domain outside the raft.
    #
    # E4: Vertical Force Balance for the raft (Newton's second law).
    #     This equation balances the forces acting on the raft in the vertical direction.
    #     It includes the integrated pressure force from the fluid (related to phi_z),
    #     the raft's inertia (related to zeta), and any external vertical forces (e.g., F_A).
    #
    # E5: Moment Balance for the raft (Newton's second law for rotation).
    #     This equation balances the moments acting on the raft about its center.
    #     It includes the integrated moment from the fluid pressure, the raft's rotational inertia
    #     (related to theta), and any external moments (e.g., from F_A at x_A).
    #
    # The right-hand side vector 'b' contains any constant forcing terms or known values.
    
    # Equation 1: Bernoulli equation on free-surface regions (left and right)
    # E1_Phi
    # Python: jnp.hstack([C13 * jnp.eye(L), jnp.zeros((L, p-L))])
    
    E1_Phi_L = hcat(C13 * Matrix{ComplexF64}(I, L, L), zeros(ComplexF64, L, p-L))
    E1_Phi_R = hcat(zeros(ComplexF64, L, p-L), C13 * Matrix{ComplexF64}(I, L, L))
    E1_Phi = vcat(E1_Phi_L, E1_Phi_R)

    # Add viscous term (second derivative) on each side of the free surface
    E1_Phi[1:L, 1:L+1] .+= C14 * d2_dx2_free
    E1_Phi[L+1:2*L, (p-L):p] .+= C14 * d2_dx2_free
    
    println("E1_Phi Shape: $(size(E1_Phi))")

    # Coefficients multiplying phi_z in Bernoulli equation
    E1_Phiz_L = hcat(C11 * Matrix{ComplexF64}(I, L, L), zeros(ComplexF64, L, p-L))
    E1_Phiz_R = hcat(zeros(ComplexF64, L, p-L), C11 * Matrix{ComplexF64}(I, L, L))
    E1_Phiz = vcat(E1_Phiz_L, E1_Phiz_R)
    
    # Add surface tension term via second derivative of phi_z
    E1_Phiz[1:L, 1:L+1] .+= C12 * d2_dx2_free
    E1_Phiz[L+1:2*L, (p-L):p] .+= C12 * d2_dx2_free
    
    println("E1_Phiz Shape: $(size(E1_Phiz))")

    # Assemble E1 block over all unknowns: [phi, phi_z, eta, zeta, theta]
    E1 = hcat(E1_Phi, E1_Phiz, zeros(ComplexF64, 2*L, p + 2))
    
    # Radiation / boundary conditions via complex wavenumber
    k = dispersion_k(omega, g, 100 * L_raft, nu, sigma, rho; k0=omega^2/g)  # Complex wavenumber
    kbar = k * L_c

    # Upstream radiation condition at left boundary
    E1[1, :] .= 0.0
    E1[1, 1] = -1.0im * kbar * dx - 1.0
    E1[1, 2] = 1.0

    # Downstream radiation condition at right boundary
    E1[end, :] .= 0.0
    E1[end, end-1] = -1.0
    E1[end, end]   = -1.0im * kbar * dx + 1.0

    # Equation 2: kinematic condition on raft (eta = zeta + x * theta) enforced on contact points
    E2_Eta   = C21 * hcat(zeros(ComplexF64, n, L), Matrix{ComplexF64}(I, n, n), zeros(ComplexF64, n, L))
    E2_Zeta  = C22 * ones(ComplexF64, n, 1)
    E2_Theta = C23 * reshape(x[x_contact], n, 1)

    E2 = hcat(zeros(ComplexF64, n, 2*p), E2_Eta, E2_Zeta, E2_Theta)

    # Equation 3: global kinematic boundary condition phi_z = i * eta
    E3_Phiz = Matrix{ComplexF64}(I, p, p)
    E3_Eta  = -1.0im * Matrix{ComplexF64}(I, p, p)
    E3 = hcat(zeros(ComplexF64, p, p), E3_Phiz, E3_Eta, zeros(ComplexF64, p, 2))
    
    # Equation 4: integrated horizontal force balance (translation)
    # E4_Phi
    # Python: integral @ (C43 + d_dx_raft**2 * C46)
    # d_dx_raft is (n, n).
    # integral is (1, n).
    # C43 is scalar.
    # C46 is scalar.
    # (C43 + d_dx_raft**2 * C46) is (n, n).
    # Result is (1, n).
    
    term4 = C43 * Matrix{ComplexF64}(I, n, n) + (d_dx_raft^2) * C46
    E4_Phi_mid = integral * term4
    E4_Phi = hcat(zeros(ComplexF64, 1, L), E4_Phi_mid, zeros(ComplexF64, 1, L))
    
    # E4_Theta
    # Python: (integral @ (C45 * x[x_contact])).reshape(1, 1)
    E4_Theta = reshape(integral * (C45 * x[x_contact]), 1, 1)
    
    # E4_Zeta
    E4_Zeta = reshape(C41 .+ integral * fill(C44, n), 1, 1)

    E4 = hcat(E4_Phi, zeros(ComplexF64, 1, 2*p), E4_Zeta, E4_Theta)

    # Equation 5: integrated moment balance about raft center (rotation)
    # E5_Phi
    # Python: integral @ (x[x_contact] * (C53 + d_dx_raft**2 * C56))
    # x[x_contact] is (n,).
    term5 = (C53 * Matrix{ComplexF64}(I, n, n) + (d_dx_raft^2) * C56)
    term5_scaled = Diagonal(x[x_contact]) * term5
    
    E5_Phi_mid = integral * term5_scaled
    E5_Phi = hcat(zeros(ComplexF64, 1, L), E5_Phi_mid, zeros(ComplexF64, 1, L))
    
    E5_Theta = reshape(C51 .+ integral * (x[x_contact] .* (C55 * x[x_contact])), 1, 1)
    E5_Zeta = reshape(integral * (x[x_contact] * C54), 1, 1)

    E5 = hcat(E5_Phi, zeros(ComplexF64, 1, 2*p), E5_Zeta, E5_Theta)

    # Equation 6: DtN relation enforcing phi_z = N * phi at all p points
    E6_Phi = dtn
    E6_Phiz = -1.0 * Matrix{ComplexF64}(I, p, p)

    E6 = hcat(E6_Phi, E6_Phiz, zeros(ComplexF64, p, p), zeros(ComplexF64, p, 2))



    # Concatenate all equation blocks into full system A x = b
    A = vcat(E1, E2, E3, E4, E5, E6)

    # Right-hand side
    b = vcat(zeros(ComplexF64, 3*p, 1), C42, C52)

    println("A shape: $(size(A))")
    println("b shape: $(size(b))")

    # Solve linear system
    # solution = A \ b
    solution = solve_tensor_system(A, b)

    # Extract variables
    phi = solution[1:p]
    eta = solution[2*p+1 : 3*p]
    zeta = solution[3*p+1]
    theta = solution[3*p+2]

    return phi, eta, zeta, theta, A
end

# Main block
if abspath(PROGRAM_FILE) == @__FILE__
    rigidSolver(1000.0, 10.0, 1e-6, 9.81, 0.05, 0.25, 0.072, 0.02, 1.0, 41)
end

end # module
