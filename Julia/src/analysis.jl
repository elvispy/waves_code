module Analysis

export beam_asymmetry,
       beam_edge_metrics,
       default_uncoupled_motor_position_EI_sweep,
       symmetric_antisymmetric_ratio,
       extract_lowest_beam_curve

"""
    beam_asymmetry(eta_left, eta_right)

Beam-end asymmetry factor

`alpha_beam = -( |eta_left|^2 - |eta_right|^2 ) / ( |eta_left|^2 + |eta_right|^2 )`
"""
function beam_asymmetry(eta_left, eta_right)
    denom = abs2(eta_left) + abs2(eta_right)
    return -(abs2(eta_left) - abs2(eta_right)) / denom
end

"""
    beam_edge_metrics(result)

Extract beam-end amplitudes and ratios from a Julia `FlexibleResult`.
"""
function beam_edge_metrics(result)
    contact = collect(Bool.(result.metadata.args.x_contact))
    idx = findall(contact)
    isempty(idx) && error("No raft-contact nodes were found in result.metadata.args.x_contact")
    left_idx = first(idx)
    right_idx = last(idx)
    eta_left_beam = result.eta[left_idx]
    eta_right_beam = result.eta[right_idx]
    eta_left_domain = first(result.eta)
    eta_right_domain = last(result.eta)
    return (
        eta_left_beam = eta_left_beam,
        eta_right_beam = eta_right_beam,
        eta_left_domain = eta_left_domain,
        eta_right_domain = eta_right_domain,
        eta_beam_ratio = abs(eta_left_beam) / max(eps(), abs(eta_right_beam)),
        eta_domain_ratio = abs(eta_left_domain) / max(eps(), abs(eta_right_domain)),
    )
end

"""
    default_uncoupled_motor_position_EI_sweep()

Return the Julia-native counterpart of the MATLAB uncoupled `x_M`-`EI` sweep
definition used for the beam-end white-curve analysis.
"""
function default_uncoupled_motor_position_EI_sweep()
    L_raft = 0.05
    base_EI = 3.0e9 * 3e-2 * (9.9e-4)^3 / 12
    base_params = (
        sigma = 72.2e-3,
        rho = 1000.0,
        nu = 0.0,
        g = 9.81,
        L_raft = L_raft,
        motor_position = 0.24 * L_raft / 2,
        d = 0.0,
        EI = base_EI,
        rho_raft = 0.052,
        domain_depth = nothing,
        L_domain = nothing,
        n = nothing,
        M = nothing,
        motor_inertia = 0.13e-3 * 2.5e-3,
        bc = :radiative,
        omega = 2 * π * 80,
        ooa = 4,
    )
    motor_position_list = collect((0.00:0.02:0.48) .* L_raft)
    EI_list = base_EI .* 10 .^ collect(range(-3, 1; length=57))
    return (; base_params, motor_position_list, EI_list)
end

"""
    symmetric_antisymmetric_ratio(eta_left, eta_right)

`log10(|S| / (|A| + eps()))` with `S = (eta_right + eta_left) / 2` and
`A = (eta_right - eta_left) / 2`.
"""
function symmetric_antisymmetric_ratio(eta_left, eta_right)
    S = (eta_right + eta_left) / 2
    A = (eta_right - eta_left) / 2
    return log10(abs(S) / (abs(A) + eps()))
end

"""
    extract_lowest_beam_curve(mp_norm_list, EI_list, eta_left, eta_right)

Reproduce the MATLAB branch extraction rule used in the beam-end modal
decomposition scripts. For each `EI` column, all `alpha_beam = 0` crossings
are linearly interpolated in `x_M / L`, filtered to the `S ~ 0` branch by
requiring `SA_ratio < 0`, and the lowest surviving crossing is retained.
"""
function extract_lowest_beam_curve(
    mp_norm_list::AbstractVector{<:Real},
    EI_list::AbstractVector{<:Real},
    eta_left::AbstractMatrix,
    eta_right::AbstractMatrix,
)
    size(eta_left) == size(eta_right) || throw(DimensionMismatch("eta_left and eta_right must match"))
    size(eta_left, 1) == length(mp_norm_list) || throw(DimensionMismatch("row count must match mp grid"))
    size(eta_left, 2) == length(EI_list) || throw(DimensionMismatch("column count must match EI grid"))

    asymmetry = beam_asymmetry.(eta_left, eta_right)
    S_grid = (eta_right .+ eta_left) ./ 2
    A_grid = (eta_right .- eta_left) ./ 2
    SA_ratio = log10.(abs.(S_grid) ./ (abs.(A_grid) .+ eps()))

    curve_EI = Float64[]
    curve_mp = Float64[]
    for ie in eachindex(EI_list)
        col = asymmetry[:, ie]
        crossings_mp = Float64[]
        crossings_SA = Float64[]
        for im in 1:(length(mp_norm_list) - 1)
            if col[im] * col[im + 1] < 0
                t = col[im] / (col[im] - col[im + 1])
                mp_zero = mp_norm_list[im] + t * (mp_norm_list[im + 1] - mp_norm_list[im])
                sa_col = SA_ratio[:, ie]
                sa_zero = sa_col[im] + t * (sa_col[im + 1] - sa_col[im])
                push!(crossings_mp, mp_zero)
                push!(crossings_SA, sa_zero)
            end
        end
        if !isempty(crossings_mp)
            mask = crossings_SA .< 0
            if any(mask)
                push!(curve_EI, EI_list[ie])
                push!(curve_mp, minimum(crossings_mp[mask]))
            end
        end
    end
    return curve_EI, curve_mp, asymmetry, SA_ratio
end

end
