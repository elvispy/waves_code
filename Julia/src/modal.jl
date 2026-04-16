module Modal

using LinearAlgebra
using Printf

export ModalDecomposition,
       decompose_raft_freefree_modes,
       trapz_weights,
       freefree_betaL_roots,
       freefree_mode_shape,
       weighted_mgs

"""
    ModalDecomposition

Container for the free-free modal projection of a raft displacement field.

The fields mirror the MATLAB helper `decompose_raft_freefree_modes`:

- `n`: retained mode labels (`0`, `1`, ...)
- `mode_type`: `"rigid"` or `"elastic"` for each retained mode
- `betaL`, `beta`: nondimensional and dimensional modal wavenumbers
- `q`: modal coefficients of the raft displacement
- `Q`, `F`: modal coefficients of `d * pressure` and forcing load
- `balance_residual`: per-mode beam-balance mismatch
- `energy_frac`: normalized `|q_n|^2`
- `eta_recon`: reconstructed raft displacement
- `recon_rel_err`: weighted relative reconstruction error
- `x_raft`: raft-node coordinates
- `Psi`: weighted-orthonormal basis on the raft grid
- `gram_cond`: condition number of `Psi' W Psi`
"""
struct ModalDecomposition
    n::Vector{Int}
    mode_type::Vector{String}
    betaL::Vector{Float64}
    beta::Vector{Float64}
    q::Vector{ComplexF64}
    Q::Vector{ComplexF64}
    F::Vector{ComplexF64}
    balance_residual::Vector{ComplexF64}
    energy_frac::Vector{Float64}
    eta_recon::Vector{ComplexF64}
    recon_rel_err::Float64
    x_raft::Vector{Float64}
    Psi::Matrix{Float64}
    gram_cond::Float64
end

"""
    trapz_weights(x)

Return vector weights such that `sum(trapz_weights(x) .* f)` matches
trapezoidal quadrature on the same grid.
"""
function trapz_weights(x::AbstractVector{<:Real})
    xvec = collect(float.(x))
    n = length(xvec)
    if n == 1
        return ones(Float64, 1)
    end
    dx = diff(xvec)
    any(dx .<= 0) && error("x_raft must be strictly increasing for trapz weights.")
    w = zeros(Float64, n)
    w[1] = dx[1] / 2
    w[end] = dx[end] / 2
    if n > 2
        @inbounds for i in 2:(n - 1)
            w[i] = (xvec[i + 1] - xvec[i - 1]) / 2
        end
    end
    return w
end

"""
    freefree_betaL_roots(n)

Positive elastic roots of `cosh(betaL) * cos(betaL) = 1`.
"""
function freefree_betaL_roots(n::Integer)
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    roots = zeros(Float64, n)
    f(y) = cosh(y) * cos(y) - 1
    for k in 1:n
        a = k * pi
        b = (k + 1) * pi
        roots[k] = find_zero_bisection(f, a, b)
    end
    return roots
end

"""
    freefree_mode_shape(xi, L, betaL)

Free-free elastic mode shape evaluated on `xi in [0, L]`.
"""
function freefree_mode_shape(xi::AbstractVector{<:Real}, L::Real, betaL::Real)
    beta = betaL / L
    bx = beta .* collect(float.(xi))
    alpha = (sin(betaL) - sinh(betaL)) / (cosh(betaL) - cos(betaL))
    psi = (sin.(bx) .+ sinh.(bx)) .+ alpha .* (cos.(bx) .+ cosh.(bx))
    scale = maximum(abs.(psi))
    return isfinite(scale) && scale > 0 ? psi ./ scale : psi
end

"""
    weighted_mgs(Phi, w)

Weighted modified Gram-Schmidt with inner product
`<u, v>_W = u' * (v .* w)`.
"""
function weighted_mgs(Phi::AbstractMatrix{<:Real}, w::AbstractVector{<:Real})
    nrow, ncol = size(Phi)
    length(w) == nrow || throw(DimensionMismatch("weight vector length must match Phi rows"))
    wvec = collect(float.(w))
    phi = Matrix{Float64}(Phi)
    psi = zeros(Float64, nrow, ncol)
    keep = falses(ncol)
    count = 0

    col_norms = sqrt.(max.(real.(sum(phi .* (phi .* wvec), dims=1)), 0))
    tol = 1e-10 * max(1.0, maximum(col_norms))

    for j in 1:ncol
        v = copy(view(phi, :, j))
        for k in 1:count
            proj = dot(view(psi, :, k), v .* wvec)
            @views v .-= psi[:, k] .* proj
        end
        nv = sqrt(max(real(dot(v, v .* wvec)), 0.0))
        if isfinite(nv) && nv > tol
            count += 1
            @views psi[:, count] .= v ./ nv
            keep[j] = true
        end
    end

    return psi[:, 1:count], keep
end

"""
    decompose_raft_freefree_modes(x, eta, pressure, loads, args; num_modes=8, include_rigid=true, verbose=true)

Project the raft displacement onto a free-free beam basis.

This ports the MATLAB helper `decompose_raft_freefree_modes`. The caller must
provide the full surface `x`/`eta` arrays plus raft-only `pressure` and
`loads`, and the `args` tuple returned by the Julia solver.
"""
function decompose_raft_freefree_modes(
    x::AbstractVector,
    eta::AbstractVector,
    pressure::AbstractVector,
    loads::AbstractVector,
    args;
    num_modes::Int=8,
    include_rigid::Bool=true,
    verbose::Bool=true,
)
    contact = collect(Bool.(args.x_contact))
    x_all = collect(float.(x))
    eta_all = ComplexF64.(eta)
    length(contact) == length(x_all) || throw(DimensionMismatch("x_contact and x length mismatch"))
    length(eta_all) == length(x_all) || throw(DimensionMismatch("eta and x length mismatch"))

    x_raft = x_all[contact]
    eta_raft = eta_all[contact]
    Nr = length(x_raft)
    Nr >= 3 || error("Need at least 3 raft points for modal decomposition.")

    p_raft = ComplexF64.(pressure)
    f_raft = ComplexF64.(loads)
    length(p_raft) == Nr || throw(DimensionMismatch("pressure must have one value per raft point"))
    length(f_raft) == Nr || throw(DimensionMismatch("loads must have one value per raft point"))

    w = trapz_weights(x_raft)
    Weta = eta_raft .* w
    Wdp = (args.d .* p_raft) .* w
    Wf = f_raft .* w

    n_requested = min(num_modes, Nr)
    n_rigid = include_rigid ? min(2, n_requested) : 0
    n_elastic = n_requested - n_rigid

    xi = x_raft .+ args.L_raft / 2
    phi_raw = zeros(Float64, Nr, n_requested)
    n_list = collect(0:(n_requested - 1))
    mode_type = fill("", n_requested)
    betaL = zeros(Float64, n_requested)
    beta = zeros(Float64, n_requested)

    col = 0
    if n_rigid >= 1
        col += 1
        phi_raw[:, col] .= 1.0
        mode_type[col] = "rigid"
    end
    if n_rigid >= 2
        col += 1
        phi_raw[:, col] .= xi .- args.L_raft / 2
        mode_type[col] = "rigid"
    end
    if n_elastic > 0
        betaL_el = freefree_betaL_roots(n_elastic)
        for j in 1:n_elastic
            col += 1
            betaL[col] = betaL_el[j]
            beta[col] = betaL[col] / args.L_raft
            phi_raw[:, col] .= freefree_mode_shape(xi, args.L_raft, betaL[col])
            mode_type[col] = "elastic"
        end
    end

    Psi, keep = weighted_mgs(phi_raw, w)
    isempty(Psi) && error("No valid modes remained after orthonormalization.")

    n_list = n_list[keep]
    mode_type = mode_type[keep]
    betaL = betaL[keep]
    beta = beta[keep]

    M = Psi' * (Psi .* w)
    rhs_q = Psi' * Weta
    rhs_Q = Psi' * Wdp
    rhs_F = Psi' * Wf

    gram_cond = cond(M)
    use_pinv = !isfinite(gram_cond) || gram_cond > 1e10
    solver = use_pinv ? pinv(M) : M
    q = use_pinv ? solver * rhs_q : solver \ rhs_q
    Q = use_pinv ? solver * rhs_Q : solver \ rhs_Q
    F = use_pinv ? solver * rhs_F : solver \ rhs_F

    beta4 = beta .^ 4
    balance_residual = (args.EI .* beta4 .- args.rho_raft .* args.omega^2) .* q .- (Q .- F)
    eta_recon = Psi * q

    recon_num = sqrt(max(real(dot(eta_raft - eta_recon, (eta_raft - eta_recon) .* w)), 0.0))
    recon_den = sqrt(max(real(dot(eta_raft, eta_raft .* w)), 0.0))
    recon_rel_err = recon_den > 0 ? recon_num / recon_den : NaN

    q_energy = abs2.(q)
    q_energy_sum = sum(q_energy)
    energy_frac = q_energy_sum > 0 ? q_energy ./ q_energy_sum : zeros(Float64, length(q))

    modal = ModalDecomposition(
        collect(Int.(n_list)),
        collect(mode_type),
        collect(betaL),
        collect(beta),
        collect(ComplexF64.(q)),
        collect(ComplexF64.(Q)),
        collect(ComplexF64.(F)),
        collect(ComplexF64.(balance_residual)),
        collect(Float64.(energy_frac)),
        collect(ComplexF64.(eta_recon)),
        recon_rel_err,
        collect(Float64.(x_raft)),
        Matrix{Float64}(Psi),
        gram_cond,
    )

    verbose && print_modal_log(modal)
    return modal
end

function decompose_raft_freefree_modes(result; kwargs...)
    args = result.metadata.args
    return decompose_raft_freefree_modes(result.x, result.eta, args.pressure, args.loads, args; kwargs...)
end

function find_zero_bisection(f, a::Real, b::Real; maxiter::Int=200, atol::Float64=1e-12)
    fa = f(a)
    fb = f(b)
    fa == 0 && return float(a)
    fb == 0 && return float(b)
    signbit(fa) == signbit(fb) && error("Root is not bracketed in [$a, $b].")

    left = float(a)
    right = float(b)
    fleft = fa
    for _ in 1:maxiter
        mid = (left + right) / 2
        fmid = f(mid)
        if abs(fmid) <= atol || abs(right - left) <= atol
            return mid
        end
        if signbit(fleft) == signbit(fmid)
            left = mid
            fleft = fmid
        else
            right = mid
        end
    end
    return (left + right) / 2
end

function print_modal_log(modal::ModalDecomposition)
    println()
    println("Modal decomposition (free-free beam basis)")
    println("n | type    | betaL      | |q_n|      | arg(q_n)   | |Q_n|      | |F_n|      | |r_n|")
    println("----------------------------------------------------------------------------------------")
    for i in eachindex(modal.n)
        println(
            lpad(modal.n[i], 1), " | ",
            rpad(modal.mode_type[i], 7), " | ",
            lpad(@sprintf("%.5f", modal.betaL[i]), 10), " | ",
            lpad(@sprintf("%.3e", abs(modal.q[i])), 10), " | ",
            lpad(@sprintf("%.3f", angle(modal.q[i])), 10), " | ",
            lpad(@sprintf("%.3e", abs(modal.Q[i])), 10), " | ",
            lpad(@sprintf("%.3e", abs(modal.F[i])), 10), " | ",
            lpad(@sprintf("%.3e", abs(modal.balance_residual[i])), 10),
        )
    end
    println(
        "Diagnostics: sum(energy_frac)=",
        @sprintf("%.6f", sum(modal.energy_frac)),
        ", recon_rel_err=", @sprintf("%.3e", modal.recon_rel_err),
        ", cond(Psi'W Psi)=", @sprintf("%.3e", modal.gram_cond),
    )
end

end
