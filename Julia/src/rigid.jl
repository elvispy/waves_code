module Rigid

using LinearAlgebra
using SparseArrays
using FiniteDifferences
using Printf

# Include our local modules
include("DtN.jl")
include("integration.jl")
include("utils.jl")

using .DtN
using .Integration
using .Utils

export rigidSolver

"""
    Diff(grid; acc=2, derivative=1)

Construct a sparse differentiation matrix for a given grid.

# Arguments
- `grid`: Vector of grid point coordinates.
- `acc`: Desired order of accuracy (default: 2).
- `derivative`: Order of the derivative to compute (default: 1).

# Returns
- A sparse differentiation matrix.
"""
function Diff(grid::AbstractVector{Float64}; acc::Int=2, derivative::Int=1)
    N = length(grid)
    dxs = diff(grid)
    
    if !all(x -> isapprox(x, dxs[1], atol=1e-6), dxs)
        @warn "Grid appears non-uniform. Using average spacing."
    end
    
    dx = sum(dxs) / length(dxs)
    
    # Interior method
    m_central = central_fdm(acc, derivative)
    grid_central = m_central.grid
    coefs_central = m_central.coefs
    
    # Boundary methods
    pts_bound = acc + derivative
    m_forward = forward_fdm(pts_bound, derivative)
    m_backward = backward_fdm(pts_bound, derivative)
    
    D = spzeros(Float64, N, N)
    
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

"""
    rigidSolver(rho, omega, nu, g, L_raft, L_domain, sigma, x_A, F_A, n)

Solve the interaction problem between a rigid raft and a viscous fluid.

# Arguments
- `rho`: Fluid density.
- `omega`: Angular frequency of forcing.
- `nu`: Kinematic viscosity.
- `g`: Gravitational acceleration.
- `L_raft`: Length of the raft.
- `L_domain`: Length of the fluid domain.
- `sigma`: Surface tension.
- `x_A`: Horizontal position of the actuator forcing.
- `F_A`: Amplitude of the vertical forcing.
- `n`: Number of grid points on the raft.

# Returns
- A tuple `(phi, eta, zeta, theta, A)` containing the solution fields and system matrix.
"""
function rigidSolver(rho, omega, nu, g, L_raft, L_domain, sigma, x_A, F_A, n)
    # Derived dimensional Parameters (SI Units)
    L_c = L_raft                   # characteristic length taken as raft length
    m_c = rho * L_c^2             # characteristic mass scale

    # Raft / domain discretization in non-dimensional units
    L_raft_adim = L_raft / L_c
    L_domain_adim = floor(L_domain / L_c) - (floor(L_domain / L_c) % 2) + 1
    p = Int(n * L_domain_adim / L_raft_adim)  # total number of grid points
    
    # Equation setup
    dtn = DtN_generator(p)         # DtN operator
    dx = 1 / n                     # non-dimensional spacing on the raft
    dtn = dtn / (L_raft / n)
    
    # Splitting domain into parts
    L = div(p - n, 2)
    integral = reshape(simpson_weights(n, Float64(dx)), 1, n)

    # Bernoulli and kinematic coefficients
    C11 = 1.0
    C12 = -(sigma / (rho * L_c^3 * omega^2)) / (g / (L_c * omega^2))
    C13 = -(omega^2 * L_c / g)      
    C14 = -(4.0im * nu / (omega * L_c^2)) / (g / (L_c * omega^2))

    C21 = 1.0   
    C22 = -1.0im 
    C23 = -1.0im 

    C41 = -m_c
    C42 = -F_A / (rho * L_c^3 * omega^2)
    C43 = 1.0im
    C44 = g / (omega^2 * L_c) 
    C45 = 0.5 * g / (omega^2 * L_c)
    C46 = 2 * nu / (omega * L_c^2)

    C51 = -(1/12) * m_c
    C52 = -(x_A / L_c) * (F_A / (rho * L_c^3 * omega^2))
    C53 = 1.0im
    C54 = (g / (omega^2 * L_c))
    C55 = 0.5 * (g / (omega^2 * L_c))
    C56 = (2 * nu / (omega * L_c^2))

    x = range(-L_domain_adim/2, stop=L_domain_adim/2, length=p)
    x_contact = abs.(x) .<= L_raft_adim/2
    H = sum(x_contact)
    left_raft_boundary = div(p - H, 2) + 1
    right_raft_boundary = left_raft_boundary + H - 1
  
    grid_x = x[left_raft_boundary] - x[left_raft_boundary-1]
    grid_free_left = collect(x[1:left_raft_boundary])
    
    D1_free = Diff(grid_free_left, acc=2, derivative=1)
    D2_free_sq = D1_free^2
    d2_dx2_free = D2_free_sq[1:end-1, :]
    
    grid_raft = collect(x[left_raft_boundary:right_raft_boundary])
    d_dx_raft = Diff(grid_raft, acc=2, derivative=1)

    E1_Phi_L = hcat(C13 * Matrix{ComplexF64}(I, L, L), zeros(ComplexF64, L, p-L))
    E1_Phi_R = hcat(zeros(ComplexF64, L, p-L), C13 * Matrix{ComplexF64}(I, L, L))
    E1_Phi = vcat(E1_Phi_L, E1_Phi_R)
    E1_Phi[1:L, 1:L+1] .+= C14 * d2_dx2_free
    E1_Phi[L+1:2*L, (p-L):p] .+= C14 * d2_dx2_free

    E1_Phiz_L = hcat(C11 * Matrix{ComplexF64}(I, L, L), zeros(ComplexF64, L, p-L))
    E1_Phiz_R = hcat(zeros(ComplexF64, L, p-L), C11 * Matrix{ComplexF64}(I, L, L))
    E1_Phiz = vcat(E1_Phiz_L, E1_Phiz_R)
    E1_Phiz[1:L, 1:L+1] .+= C12 * d2_dx2_free
    E1_Phiz[L+1:2*L, (p-L):p] .+= C12 * d2_dx2_free

    E1 = hcat(E1_Phi, E1_Phiz, zeros(ComplexF64, 2*L, p + 2))
    
    k = dispersion_k(omega, g, 100 * L_raft, nu, sigma, rho; k0=omega^2/g)
    kbar = k * L_c

    E1[1, 1] = -1.0im * kbar * dx - 1.0
    E1[1, 2] = 1.0
    E1[end, end-1] = -1.0
    E1[end, end]   = -1.0im * kbar * dx + 1.0

    E2_Eta   = C21 * hcat(zeros(ComplexF64, n, L), Matrix{ComplexF64}(I, n, n), zeros(ComplexF64, n, L))
    E2_Zeta  = C22 * ones(ComplexF64, n, 1)
    E2_Theta = C23 * reshape(x[x_contact], n, 1)
    E2 = hcat(zeros(ComplexF64, n, 2*p), E2_Eta, E2_Zeta, E2_Theta)

    E3_Phiz = Matrix{ComplexF64}(I, p, p)
    E3_Eta  = -1.0im * Matrix{ComplexF64}(I, p, p)
    E3 = hcat(zeros(ComplexF64, p, p), E3_Phiz, E3_Eta, zeros(ComplexF64, p, 2))
    
    term4 = C43 * Matrix{ComplexF64}(I, n, n) + (d_dx_raft^2) * C46
    E4_Phi_mid = integral * term4
    E4_Phi = hcat(zeros(ComplexF64, 1, L), E4_Phi_mid, zeros(ComplexF64, 1, L))
    E4_Theta = reshape(integral * (C45 * x[x_contact]), 1, 1)
    E4_Zeta = reshape(C41 .+ integral * fill(C44, n), 1, 1)
    E4 = hcat(E4_Phi, zeros(ComplexF64, 1, 2*p), E4_Zeta, E4_Theta)

    term5 = (C53 * Matrix{ComplexF64}(I, n, n) + (d_dx_raft^2) * C56)
    term5_scaled = Diagonal(x[x_contact]) * term5
    E5_Phi_mid = integral * term5_scaled
    E5_Phi = hcat(zeros(ComplexF64, 1, L), E5_Phi_mid, zeros(ComplexF64, 1, L))
    E5_Theta = reshape(C51 .+ integral * (x[x_contact] .* (C55 * x[x_contact])), 1, 1)
    E5_Zeta = reshape(integral * (x[x_contact] * C54), 1, 1)
    E5 = hcat(E5_Phi, zeros(ComplexF64, 1, 2*p), E5_Zeta, E5_Theta)

    E6_Phi = dtn
    E6_Phiz = -1.0 * Matrix{ComplexF64}(I, p, p)
    E6 = hcat(E6_Phi, E6_Phiz, zeros(ComplexF64, p, p), zeros(ComplexF64, p, 2))

    A = vcat(E1, E2, E3, E4, E5, E6)
    b = vcat(zeros(ComplexF64, 3*p, 1), C42, C52)
    solution = solve_tensor_system(A, b)

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
