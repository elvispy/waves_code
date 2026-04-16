module Surferbot

using SparseArrays
using LinearAlgebra

include("fd.jl")
include("integration.jl")
include("analysis.jl")
include("modal.jl")
include("postprocess.jl")
include("sweep.jl")
include("utils.jl")

using .Analysis: beam_asymmetry,
                 beam_edge_metrics,
                 default_uncoupled_motor_position_EI_sweep,
                 symmetric_antisymmetric_ratio,
                 extract_lowest_beam_curve
using .FD: getNonCompactFDMWeights, getNonCompactFDmatrix, getNonCompactFDmatrix2D
using .Integration: simpson_weights
using .Modal: ModalDecomposition,
              decompose_raft_freefree_modes,
              trapz_weights,
              freefree_betaL_roots,
              freefree_mode_shape,
              weighted_mgs
using .PostProcess: calculate_surferbot_outputs
using .Sweep: SweepSummary,
              SweepArtifact,
              apply_parameter_overrides,
              expand_parameter_grid,
              summarize_result,
              sweep_parameters,
              save_sweep,
              load_sweep
using .Utils: dispersion_k, gaussian_load, solve_tensor_system

export FlexibleParams,
       FlexibleResult,
       getNonCompactFDMWeights,
       getNonCompactFDmatrix,
       getNonCompactFDmatrix2D,
       simpson_weights,
       beam_asymmetry,
       beam_edge_metrics,
       default_uncoupled_motor_position_EI_sweep,
       symmetric_antisymmetric_ratio,
       extract_lowest_beam_curve,
       ModalDecomposition,
       decompose_raft_freefree_modes,
       trapz_weights,
       freefree_betaL_roots,
       freefree_mode_shape,
       weighted_mgs,
       SweepSummary,
       SweepArtifact,
       apply_parameter_overrides,
       expand_parameter_grid,
       summarize_result,
       sweep_parameters,
       save_sweep,
       load_sweep,
       dispersion_k,
       gaussian_load,
       solve_tensor_system,
       derive_params,
       assemble_flexible_system,
       flexible_solver,
       flexible_surferbot_v2_julia

"""
    FlexibleParams

Physical, geometric, and numerical parameters for the flexible Surferbot
solver.

This struct mirrors the MATLAB entry point `flexible_surferbot_v2` while giving
the parameters explicit names and types. The most important groups are:

- fluid properties: `sigma`, `rho`, `nu`, `g`
- forcing: `omega`, `motor_position`, `motor_force`, `forcing_width`
- raft properties: `L_raft`, `d`, `EI`, `rho_raft`
- numerical resolution: `n`, `M`, `ooa`, `L_domain`, `domain_depth`
- boundary condition: `bc`
"""
Base.@kwdef struct FlexibleParams
    sigma::Float64 = 72.2e-3
    rho::Float64 = 1000.0
    omega::Float64 = 2 * pi * 10.0
    nu::Float64 = 1e-6
    g::Float64 = 9.81
    L_raft::Float64 = 0.05
    motor_position::Float64 = 0.6 / 2.5 * 0.05
    d::Union{Nothing, Float64} = nothing
    EI::Float64 = 3.0e9 * 3e-2 * 9e-4^3 / 12
    rho_raft::Float64 = 0.052
    L_domain::Union{Nothing, Float64} = nothing
    domain_depth::Union{Nothing, Float64} = nothing
    n::Union{Nothing, Int} = nothing
    M::Union{Nothing, Int} = nothing
    ooa::Int = 4
    motor_inertia::Float64 = 0.13e-3 * 2.5e-3
    motor_force::Union{Nothing, Float64} = nothing
    forcing_width::Float64 = 0.05
    bc::Symbol = :radiative
end

"""
    FlexibleResult

Primary outputs of the flexible Surferbot solve.

- `U`: drift speed inferred from the drag-law closure
- `power`: mean actuator power returned by the current sign convention
- `thrust`: mean wave-driven thrust
- `phi`, `phi_z`: harmonic potential fields
- `eta`, `pressure`: reconstructed free-surface elevation and raft pressure

`metadata` stores the dimensional arguments used by the postprocessing layer
and, optionally, the assembled linear system when `return_system=true`.
"""
struct FlexibleResult
    U::Float64
    power::Float64
    thrust::Float64
    x::Vector{Float64}
    z::Vector{Float64}
    phi::Matrix{ComplexF64}
    phi_z::Matrix{ComplexF64}
    eta::Vector{ComplexF64}
    pressure::Vector{ComplexF64}
    metadata::NamedTuple
end

"""
    derive_params(params)

Construct all derived scales, nondimensional groups, grids, and distributed
loads needed by the linear solver.

This is the Julia analogue of the MATLAB preprocessing logic before system
assembly. In particular, it computes:

- characteristic scales `L_c`, `t_c`, `m_c`, `F_c`
- nondimensional groups `Gamma`, `Fr`, `Re`, `kappa`, `We`, `Lambda`
- the weakly viscous gravity-capillary wavenumber `k`
- the numerical grid in `x` and `z`
- the raft-contact mask and the free-surface mask
- the distributed motor forcing on the contact region
"""
function derive_params(params::FlexibleParams)
    motor_force = isnothing(params.motor_force) ? params.motor_inertia * params.omega^2 : params.motor_force
    d = isnothing(params.d) ? 0.6 * params.L_raft : params.d
    motor_position = clamp(params.motor_position, -params.L_raft / 2, params.L_raft / 2)

    L_c = params.L_raft
    t_c = 1 / params.omega
    m_c = params.rho_raft * L_c
    F_c = m_c * L_c / t_c^2

    nd_groups = (
        Gamma = params.rho * params.L_raft^2 / params.rho_raft,
        Fr = sqrt(params.L_raft * params.omega^2 / params.g),
        Re = params.L_raft^2 * params.omega / params.nu,
        kappa = params.EI / (params.rho_raft * params.L_raft^4 * params.omega^2),
        We = params.rho_raft * params.L_raft * params.omega^2 / params.sigma,
        Lambda = d / params.L_raft,
    )

    domain_depth = isnothing(params.domain_depth) ? NaN : params.domain_depth
    k = ComplexF64(NaN + NaN * im)
    while isnan(real(k)) || tanh(real(k) * domain_depth) < 0.99
        k = ComplexF64(dispersion_k(params.omega, params.g, domain_depth, params.nu, params.sigma, params.rho))
        if isnan(domain_depth)
            domain_depth = 2.5 * params.g / params.omega^2
        else
            domain_depth *= 1.05
        end
    end

    res = 80
    k_real = real(k)
    n = if isnothing(params.n)
        n_guess = max(res, ceil(Int, res / (2 * pi / k_real) * params.L_raft))
        n_guess + mod(n_guess, 2) + 1
    else
        params.n
    end

    M = isnothing(params.M) ? ceil(Int, res * k_real * domain_depth) : params.M
    L_domain = if isnothing(params.L_domain)
        min(3 * params.L_raft, round(20 * 2 * pi / k_real + params.L_raft; sigdigits=2))
    else
        params.L_domain
    end

    N = round(Int, (n - 1) * L_domain / params.L_raft) + 1
    L_domain_adim = N / n

    x = collect(range(-L_domain_adim / 2, stop = L_domain_adim / 2, length = N))
    dx = x[2] - x[1]

    z = collect(range(-domain_depth, stop = 0.0, length = M)) ./ L_c
    dz = abs(z[2] - z[1])

    x_contact = abs.(x) .<= params.L_raft / (2 * L_c)
    x_free = abs.(x) .> params.L_raft / (2 * L_c)
    x_free[1] = false
    x_free[end] = false

    loads = motor_force / F_c * gaussian_load(motor_position / L_c, params.forcing_width, x[x_contact])

    return (
        params = params,
        motor_force = motor_force,
        d = d,
        motor_position = motor_position,
        L_c = L_c,
        t_c = t_c,
        m_c = m_c,
        F_c = F_c,
        nd_groups = nd_groups,
        domain_depth = domain_depth,
        k = k,
        n = n,
        M = M,
        L_domain = L_domain,
        N = N,
        x = x,
        z = z,
        dx = dx,
        dz = dz,
        x_contact = x_contact,
        x_free = x_free,
        loads = loads,
    )
end

"""
    assemble_flexible_system(params)

Assemble the sparse harmonic linear system for the coupled fluid-raft problem.

The unknown vector is ordered as

`[phi; phi_z; M]`

where:

- `phi` is the harmonic velocity potential on the 2D grid
- `phi_z` is its vertical derivative
- `M` is the raft bending-moment variable on the contact region

The block matrix contains:

- Bernoulli/free-surface rows on the uncovered interface
- Euler-Bernoulli beam rows on the raft-contact region
- Laplace rows in the fluid interior
- impermeability rows at the bottom
- side-wall or radiative boundary rows in `x`
- derivative-constraint rows enforcing `phi_z - Dz*phi = 0`
- bending-moment closure rows on the raft

This is the Julia counterpart of MATLAB `build_system_v2.m`.
"""
function assemble_flexible_system(params::FlexibleParams)
    derived = derive_params(params)
    NP = derived.N * derived.M
    nb_contact = count(derived.x_contact)
    system_size = 2 * NP + nb_contact
    BCtype = String(derived.params.bc)
    Fr = derived.nd_groups.Fr
    Gamma = derived.nd_groups.Gamma
    We = derived.nd_groups.We
    kappa = derived.nd_groups.kappa
    Lambda = derived.nd_groups.Lambda
    Re = derived.nd_groups.Re
    I_NP = sparse(I, NP, NP)
    I_CC = sparse(I, nb_contact, nb_contact)

    topMask = falses(derived.M, derived.N)
    topMask[end, 2:(end - 1)] .= true
    freeMask = repeat(reshape(derived.x_free, 1, derived.N), derived.M, 1) .& topMask
    contactMask = repeat(reshape(derived.x_contact, 1, derived.N), derived.M, 1) .& topMask

    bottomMask = falses(derived.M, derived.N)
    bottomMask[1, 2:(end - 1)] .= true
    rightEdgeMask = falses(derived.M, derived.N)
    rightEdgeMask[:, end] .= true
    leftEdgeMask = falses(derived.M, derived.N)
    leftEdgeMask[:, 1] .= true
    bulkMask = falses(derived.M, derived.N)
    bulkMask[2:(end - 1), 2:(end - 1)] .= true

    idxFreeSurf = findall(vec(freeMask))
    idxContact = findall(vec(contactMask))
    idxBulk = findall(vec(bulkMask))
    idxBottom = findall(vec(bottomMask))
    idxLeftEdge = findall(vec(leftEdgeMask))
    idxRightEdge = findall(vec(rightEdgeMask))

    half_free = length(idxFreeSurf) ÷ 2
    first_contact = first(idxContact)
    last_contact = last(idxContact)
    idxLeftFreeSurf = vcat(derived.M, idxFreeSurf[1:half_free], first_contact)
    idxRightFreeSurf = vcat(last_contact, idxFreeSurf[(half_free + 1):end], NP)
    nbLeft = length(idxLeftFreeSurf)

    S11 = spzeros(ComplexF64, NP, NP)
    S12 = spzeros(ComplexF64, NP, NP)
    S13 = spzeros(ComplexF64, NP, nb_contact)
    S21 = spzeros(ComplexF64, NP, NP)
    S22 = spzeros(ComplexF64, NP, NP)
    S23 = spzeros(ComplexF64, NP, nb_contact)
    S31 = spzeros(ComplexF64, nb_contact, NP)
    S32 = spzeros(ComplexF64, nb_contact, NP)
    S33 = spzeros(ComplexF64, nb_contact, nb_contact)

    L = idxLeftFreeSurf
    DxFree = getNonCompactFDmatrix(nbLeft, 1.0, 1, derived.params.ooa)
    DxxFree = getNonCompactFDmatrix(nbLeft, 1.0, 2, derived.params.ooa)
    S11[L[2:(end - 1)], L] = derived.dx^2 .* I_NP[L[2:(end - 1)], L] .+ (4im / Re) .* DxxFree[2:(end - 1), :]
    S12[L[2:(end - 1)], L] = (-derived.dx^2 / Fr^2) .* I_NP[L[2:(end - 1)], L] .+ (1 / (We * Gamma)) .* DxxFree[2:(end - 1), :]

    R = idxRightFreeSurf
    S11[R[2:(end - 1)], R] = derived.dx^2 .* I_NP[R[2:(end - 1)], R] .+ (4im / Re) .* DxxFree[2:(end - 1), :]
    S12[R[2:(end - 1)], R] = (-derived.dx^2 / Fr^2) .* I_NP[R[2:(end - 1)], R] .+ (1 / (We * Gamma)) .* DxxFree[2:(end - 1), :]

    CC = idxContact
    DxRaft = getNonCompactFDmatrix(nb_contact, 1.0, 1, derived.params.ooa)
    Dx2Raft = getNonCompactFDmatrix(nb_contact, 1.0, 2, derived.params.ooa)

    S11[CC, CC] = (1im * Lambda * Gamma * derived.dx^2) .* I_NP[CC, CC] .+ (2 * Gamma * Lambda / Re) .* Dx2Raft
    S12[CC, CC] = ((1im - 1im * Gamma * Lambda / Fr^2) * derived.dx^2) .* I_NP[CC, CC]
    S13[CC, :] = Dx2Raft

    boundary_contact = [1, nb_contact]
    S11[CC[boundary_contact], :] .= 0
    S12[CC[boundary_contact], :] .= 0
    S13[CC[boundary_contact], :] .= 0

    S13[CC[1], :] = DxRaft[1, :]
    S12[CC[1], L] = (-1im * derived.dx * Lambda / We) .* DxFree[end, :]
    S13[CC[end], :] = DxRaft[end, :]
    S12[CC[end], R] = (-1im * derived.dx * Lambda / We) .* DxFree[1, :]

    S32[:, CC] = 1im .* Dx2Raft
    S33 .= (derived.dx^2 / kappa) .* I_CC
    S32[boundary_contact, :] .= 0
    S33[boundary_contact, :] .= 0
    S33[1, 1] = 1
    S33[end, end] = 1

    Dx, Dz = getNonCompactFDmatrix2D(derived.N, derived.M, 1.0, 1.0, 1, derived.params.ooa)
    Dxx, Dzz = getNonCompactFDmatrix2D(derived.N, derived.M, 1.0, 1.0, 2, derived.params.ooa)
    S11[idxBulk, :] = Dxx[idxBulk, :] .+ Dzz[idxBulk, :] .* (derived.dx / derived.dz)^2
    S12[idxBottom, :] = I_NP[idxBottom, :]

    S12[idxLeftEdge, :] .= 0
    S12[idxRightEdge, :] .= 0
    if startswith(lowercase(BCtype), "n")
        S11[idxLeftEdge, :] = Dx[idxLeftEdge, :]
        S11[idxRightEdge, :] = Dx[idxRightEdge, :]
    elseif startswith(lowercase(BCtype), "r")
        S11[idxLeftEdge, :] = (-1im * derived.k * derived.params.L_raft * derived.dx) .* I_NP[idxLeftEdge, :] .+ Dx[idxLeftEdge, :]
        S11[idxRightEdge, :] = (1im * derived.k * derived.params.L_raft * derived.dx) .* I_NP[idxRightEdge, :] .+ Dx[idxRightEdge, :]
    end

    S21 = Dz
    S22 = -derived.dz .* I_NP

    b = zeros(ComplexF64, system_size)
    b[CC] = -derived.dx^2 .* ComplexF64.(derived.loads)
    b[CC[boundary_contact]] .= 0

    A = [
        S11 S12 S13
        S21 S22 S23
        S31 S32 S33
    ]

    return (
        A = sparse(ComplexF64.(A)),
        b = b,
        derived = derived,
        masks = (
            topMask = topMask,
            freeMask = freeMask,
            contactMask = contactMask,
            bottomMask = bottomMask,
            leftEdgeMask = leftEdgeMask,
            rightEdgeMask = rightEdgeMask,
            bulkMask = bulkMask,
        ),
        indices = (
            idxBulk = idxBulk,
            idxBottom = idxBottom,
            idxLeftEdge = idxLeftEdge,
            idxRightEdge = idxRightEdge,
            idxContact = idxContact,
            idxLeftFreeSurf = idxLeftFreeSurf,
            idxRightFreeSurf = idxRightFreeSurf,
        ),
    )
end

"""
    flexible_solver(params; return_system=false)

Solve the coupled flexible Surferbot problem and postprocess the solution into
physically meaningful outputs.

The workflow is:

1. derive scales, grids, and distributed loads
2. assemble the sparse block system
3. solve the harmonic linear problem
4. reconstruct `phi`, `phi_z`, `eta`, and pressure
5. compute thrust, drift speed, and mean power input

If `return_system=true`, the assembled system and derived quantities are also
returned for debugging, parity testing, and optimization.
"""
function flexible_solver(params::FlexibleParams; return_system::Bool=false)
    system = assemble_flexible_system(params)
    solution = solve_tensor_system(system.A, system.b)
    NP = system.derived.N * system.derived.M
    solution = solution[1:(2 * NP)]

    phi = reshape(solution[1:NP], system.derived.M, system.derived.N)
    phi_z = reshape(solution[(NP + 1):(2 * NP)], system.derived.M, system.derived.N)

    x = system.derived.x .* system.derived.L_c
    z = system.derived.z .* system.derived.L_c

    args = (
        sigma = params.sigma,
        rho = params.rho,
        omega = params.omega,
        nu = params.nu,
        g = params.g,
        L_raft = params.L_raft,
        d = system.derived.d,
        EI = params.EI,
        rho_raft = params.rho_raft,
        nd_groups = system.derived.nd_groups,
        x_contact = system.derived.x_contact,
        x = x,
        loads = system.derived.loads .* system.derived.F_c ./ system.derived.L_c,
        N = system.derived.N,
        M = system.derived.M,
        dx = system.derived.dx .* system.derived.L_c,
        dz = system.derived.dz .* system.derived.L_c,
        t_c = system.derived.t_c,
        L_c = system.derived.L_c,
        m_c = system.derived.m_c,
        k = system.derived.k,
        ooa = params.ooa,
    )

    U, power, thrust, eta, p = calculate_surferbot_outputs(args, phi, phi_z, getNonCompactFDmatrix, getNonCompactFDmatrix2D)

    result = FlexibleResult(
        U,
        power,
        thrust,
        x,
        z,
        Matrix(phi .* system.derived.L_c^2 ./ system.derived.t_c),
        Matrix(phi_z .* system.derived.L_c ./ system.derived.t_c),
        collect(eta),
        collect(p),
        (
            args = merge(args, (pressure = p, power = power, thrust = thrust, phi_z = phi_z .* system.derived.L_c ./ system.derived.t_c)),
            system = return_system ? system : nothing,
        ),
    )

    return return_system ? (result = result, system = system) : result
end

"""
    flexible_surferbot_v2_julia(; kwargs...)

MATLAB-style compatibility wrapper around `flexible_solver`.

This preserves the feel of the original MATLAB entry point while the typed
Julia API remains the primary interface for new code.
"""
function flexible_surferbot_v2_julia(; kwargs...)
    result = flexible_solver(FlexibleParams(; kwargs...))
    args = result.metadata.args
    return result.U, result.x, result.z, result.phi, result.eta, args
end

end
