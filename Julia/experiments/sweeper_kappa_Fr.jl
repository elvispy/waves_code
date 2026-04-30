using Surferbot
using Printf
using Base.Threads
using DelimitedFiles

"""
sweeper_kappa_Fr.jl

Clean (κ, Fr) sweep at fixed ω (canonical Surferbot frequency). Only EI and g
are varied per grid point, so Re = L²ω/ν and We = ρ_R Lω²/σ stay constant
across the entire plane — the only ND parameters that move are κ and Fr.

Convention:
    κ  = EI / (ρ_R L^4 ω²)
    Fr = √(L ω² / g)
For a (κ, Fr) grid point we set
    g  = L ω² / Fr²          (effective gravity; ω held at canonical value)
    EI = κ · ρ_R · L^4 · ω²

The CSV layout matches `sweeper_modal_coefficients.jl` except the two primary
axes are now `log10_kappa` and `log10_Fr`, and the `g` column records the
effective gravity for each row (omega is constant and equals base_params.omega).
"""

function generate_header(num_modes)
    header = ["log10_kappa", "log10_Fr",
              "EI", "g", "L_raft", "d", "rho_raft", "xM_over_L", "alpha"]
    append!(header, [
        "eta_1_beam_re", "eta_1_beam_im",
        "eta_1_domain_re", "eta_1_domain_im",
        "eta_end_beam_re", "eta_end_beam_im",
        "eta_end_domain_re", "eta_end_domain_im"
    ])
    for n in 0:(num_modes - 1)
        append!(header, [
            "q_w$(n)_re", "q_w$(n)_im",
            "Q_w$(n)_re", "Q_w$(n)_im",
            "F_w$(n)_re", "F_w$(n)_im"
        ])
    end
    return join(header, ",")
end

function format_row(params, log10_kappa, log10_Fr, res, modal)
    metrics = Surferbot.Analysis.beam_edge_metrics(res)
    alpha   = Surferbot.Analysis.beam_asymmetry(metrics.eta_left_beam,
                                                metrics.eta_right_beam)

    row_data = [
        log10_kappa,
        log10_Fr,
        params.EI,
        params.g,
        params.L_raft,
        isnothing(params.d) ? 0.0 : params.d,
        params.rho_raft,
        params.motor_position / params.L_raft,
        alpha,
    ]
    append!(row_data, [
        real(metrics.eta_left_beam),  imag(metrics.eta_left_beam),
        real(metrics.eta_left_domain), imag(metrics.eta_left_domain),
        real(metrics.eta_right_beam), imag(metrics.eta_right_beam),
        real(metrics.eta_right_domain), imag(metrics.eta_right_domain),
    ])
    for i in 1:length(modal.n)
        push!(row_data, real(modal.q[i])); push!(row_data, imag(modal.q[i]))
        push!(row_data, real(modal.Q[i])); push!(row_data, imag(modal.Q[i]))
        push!(row_data, real(modal.F[i])); push!(row_data, imag(modal.F[i]))
    end
    return join([@sprintf("%.12e", x) for x in row_data], ",")
end

# Build a FlexibleParams override at a (κ, Fr) point.
# ω is fixed at its canonical value; only EI and g vary so that Re and We
# remain constant across the entire (κ, Fr) plane.
function params_at_kappa_Fr(base_params, kappa::Real, Fr::Real, L::Real,
                            rho_raft::Real, omega::Real)
    g  = L * omega^2 / Fr^2
    EI = kappa * rho_raft * L^4 * omega^2
    return Surferbot.Sweep.apply_parameter_overrides(base_params, (
        EI = EI,
        g  = g,
    ))
end

function run_sweep(output_path, is_coupled;
                   nkappa=100, nFr=100, num_modes=8, task_id=nothing,
                   log10_kappa_range = range(+1.0, -5.0; length=100),
                   log10_Fr_range    = range(+2.0,  0.0; length=100))
    # ω is fixed at the canonical Surferbot value; only EI and g vary per point.
    base_params = is_coupled ?
        Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params :
        Surferbot.Analysis.default_uncoupled_motor_position_EI_sweep().base_params

    L_raft   = base_params.L_raft
    rho_raft = base_params.rho_raft
    omega    = base_params.omega

    @assert length(log10_kappa_range) == nkappa
    @assert length(log10_Fr_range)    == nFr

    kappa_list = 10.0 .^ collect(log10_kappa_range)
    Fr_list    = 10.0 .^ collect(log10_Fr_range)

    # Each "task" walks one Fr-slice (all kappa values at fixed Fr).
    indices = isnothing(task_id) ? (1:nFr) : (task_id:task_id)

    if isnothing(task_id) || task_id == 1
        open(output_path, "w") do io
            println(io, generate_header(num_modes))
        end
    end

    write_lock = ReentrantLock()

    if isnothing(task_id)
        println("Starting full κ-Fr sweep: $(is_coupled ? "Coupled" : "Uncoupled") -> $output_path")
        println("Grid: $nFr Fr × $nkappa κ ($((nFr*nkappa)) total)")
        @threads for iFr in indices
            process_Fr_slice(iFr, Fr_list, kappa_list, base_params,
                             L_raft, rho_raft, omega, output_path, num_modes,
                             write_lock, task_id, nkappa, nFr,
                             collect(log10_Fr_range), collect(log10_kappa_range))
        end
    else
        for iFr in indices
            process_Fr_slice(iFr, Fr_list, kappa_list, base_params,
                             L_raft, rho_raft, omega, output_path, num_modes,
                             write_lock, task_id, nkappa, nFr,
                             collect(log10_Fr_range), collect(log10_kappa_range))
        end
    end
end

function process_Fr_slice(iFr, Fr_list, kappa_list, base_params,
                          L_raft, rho_raft, omega, output_path, num_modes,
                          write_lock, task_id, nkappa, nFr,
                          log10_Fr_axis, log10_kappa_axis)
    Fr = Fr_list[iFr]
    log10_Fr = log10_Fr_axis[iFr]
    rows_to_write = String[]

    for ik in 1:nkappa
        kappa       = kappa_list[ik]
        log10_kappa = log10_kappa_axis[ik]
        params      = params_at_kappa_Fr(base_params, kappa, Fr,
                                         L_raft, rho_raft, omega)
        try
            res   = flexible_solver(params)
            modal = decompose_raft_freefree_modes(res; num_modes=num_modes,
                                                       verbose=false)
            push!(rows_to_write,
                  format_row(params, log10_kappa, log10_Fr, res, modal))
        catch e
            @warn "Failed at κ=$(kappa), Fr=$(Fr): $e"
        end
    end

    lock(write_lock) do
        open(output_path, "a") do io
            for row in rows_to_write
                println(io, row)
            end
        end
    end

    if isnothing(task_id)
        @printf("Completed Fr index %3d/%3d  (log10 Fr = %.3f)\n",
                iFr, nFr, log10_Fr)
    else
        @printf("Task %d:  Fr = %.3f  (log10 Fr = %.3f)  done\n",
                task_id, Fr, log10_Fr)
    end
end

function main()
    # Usage: julia sweeper_kappa_Fr.jl [task_id] [uncoupled|coupled|both]
    task_id_str = get(ARGS, 1, "")
    task_id     = isempty(task_id_str) ? nothing : parse(Int, task_id_str)
    mode        = get(ARGS, 2, "both")

    output_dir = joinpath(@__DIR__, "..", "output", "csv")
    mkpath(output_dir)

    if mode == "both" || mode == "uncoupled"
        run_sweep(joinpath(output_dir, "sweeper_kappa_Fr_uncoupled.csv"),
                  false; task_id=task_id)
    end
    if mode == "both" || mode == "coupled"
        run_sweep(joinpath(output_dir, "sweeper_kappa_Fr_coupled.csv"),
                  true;  task_id=task_id)
    end

    isnothing(task_id) && println("\nAll κ-Fr sweeps completed.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
