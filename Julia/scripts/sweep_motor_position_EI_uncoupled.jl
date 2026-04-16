using JLD2
using Statistics
using Surferbot

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function main(
    save_dir::AbstractString=joinpath(@__DIR__, "..", "data");
    base_params_override=nothing,
    motor_position_list_override=nothing,
    EI_list_override=nothing,
    outfile::AbstractString="sweep_motorPosition_EI_uncoupled.jld2",
)
    save_dir = ensure_dir(normpath(save_dir))
    sweep = default_uncoupled_motor_position_EI_sweep()
    base = isnothing(base_params_override) ? sweep.base_params : base_params_override
    motor_position_list = isnothing(motor_position_list_override) ? sweep.motor_position_list : collect(motor_position_list_override)
    EI_list = isnothing(EI_list_override) ? sweep.EI_list : collect(EI_list_override)

    n_mp = length(motor_position_list)
    n_ei = length(EI_list)

    eta_left_beam = Matrix{ComplexF64}(undef, n_mp, n_ei)
    eta_right_beam = Matrix{ComplexF64}(undef, n_mp, n_ei)
    eta_left_domain = Matrix{ComplexF64}(undef, n_mp, n_ei)
    eta_right_domain = Matrix{ComplexF64}(undef, n_mp, n_ei)
    thrust = Matrix{Float64}(undef, n_mp, n_ei)
    power = Matrix{Float64}(undef, n_mp, n_ei)
    tail_flat_ratio = Matrix{Float64}(undef, n_mp, n_ei)

    println("Running Julia uncoupled motor-position EI sweep")
    println("grid: $(n_mp) motor positions x $(n_ei) EI values")

    for (ip, motor_position) in enumerate(motor_position_list)
        println("row $ip / $n_mp: x_M/L = $(round(motor_position / base.L_raft; digits=4))")
        for (ie, EI) in enumerate(EI_list)
            params = FlexibleParams(;
                sigma=base.sigma,
                rho=base.rho,
                nu=base.nu,
                g=base.g,
                L_raft=base.L_raft,
                motor_position=motor_position,
                d=base.d,
                EI=EI,
                rho_raft=base.rho_raft,
                domain_depth=base.domain_depth,
                L_domain=base.L_domain,
                n=base.n,
                M=base.M,
                motor_inertia=base.motor_inertia,
                bc=base.bc,
                omega=base.omega,
                ooa=base.ooa,
            )
            result = flexible_solver(params)
            metrics = beam_edge_metrics(result)

            eta_left_beam[ip, ie] = metrics.eta_left_beam
            eta_right_beam[ip, ie] = metrics.eta_right_beam
            eta_left_domain[ip, ie] = metrics.eta_left_domain
            eta_right_domain[ip, ie] = metrics.eta_right_domain
            thrust[ip, ie] = result.thrust
            power[ip, ie] = result.power

            left_count = max(1, ceil(Int, 0.05 * length(result.eta)))
            tail = abs.(result.eta[1:left_count])
            tail_flat_ratio[ip, ie] = std(tail) / max(eps(), mean(tail))
        end
    end

    alpha_beam = beam_asymmetry.(eta_left_beam, eta_right_beam)
    outfile = joinpath(save_dir, outfile)
    jldsave(
        outfile;
        base_params=base,
        motor_position_list=motor_position_list,
        EI_list=EI_list,
        eta_left_beam=eta_left_beam,
        eta_right_beam=eta_right_beam,
        eta_left_domain=eta_left_domain,
        eta_right_domain=eta_right_domain,
        thrust=thrust,
        power=power,
        tail_flat_ratio=tail_flat_ratio,
        alpha_beam=alpha_beam,
    )
    println("Saved $outfile")
    return outfile
end

if abspath(PROGRAM_FILE) == @__FILE__
    save_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data")
    main(save_dir)
end
