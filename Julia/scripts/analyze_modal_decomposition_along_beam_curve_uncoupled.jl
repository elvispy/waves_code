using JLD2
using Plots
using Surferbot

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function load_uncoupled_sweep(path::AbstractString)
    data = JLD2.load(path)
    required = (
        "base_params",
        "motor_position_list",
        "EI_list",
        "eta_left_beam",
        "eta_right_beam",
    )
    for key in required
        haskey(data, key) || error("Missing `$key` in $path. Regenerate the Julia uncoupled sweep artifact.")
    end
    return data
end

function write_modal_summary_csv(path::AbstractString, sample_EI, sample_mp, mode_types, all_q, all_energy, all_phase)
    open(path, "w") do io
        println(io, "sample_index,log10_EI,xM_over_L,mode_index,mode_type,abs_q,energy_frac,phase_deg,re_q,im_q")
        for ip in axes(all_q, 1), j in axes(all_q, 2)
            println(
                io,
                join(
                    (
                        ip,
                        log10(sample_EI[ip]),
                        sample_mp[ip],
                        j - 1,
                        mode_types[j],
                        abs(all_q[ip, j]),
                        all_energy[ip, j],
                        all_phase[ip, j],
                        real(all_q[ip, j]),
                        imag(all_q[ip, j]),
                    ),
                    ",",
                ),
            )
        end
    end
end

function main(
    data_dir::AbstractString=joinpath(@__DIR__, "..", "data");
    sweep_file::AbstractString="sweep_motorPosition_EI_uncoupled.jld2",
    n_sample::Int=12,
    n_modes::Int=8,
    pdf_file::AbstractString="analyze_modal_decomposition_along_beam_curve_uncoupled.pdf",
    csv_file::AbstractString="analyze_modal_decomposition_along_beam_curve_uncoupled.csv",
)
    data_dir = ensure_dir(normpath(data_dir))
    sweep_path = joinpath(data_dir, sweep_file)
    data = load_uncoupled_sweep(sweep_path)

    base = NamedTuple(data["base_params"])
    motor_position_list = vec(Float64.(data["motor_position_list"]))
    EI_list = vec(Float64.(data["EI_list"]))
    eta_left_beam = ComplexF64.(data["eta_left_beam"])
    eta_right_beam = ComplexF64.(data["eta_right_beam"])

    mp_norm_list = motor_position_list ./ base.L_raft
    curve_EI, curve_mp, _, _ = extract_lowest_beam_curve(mp_norm_list, EI_list, eta_left_beam, eta_right_beam)
    isempty(curve_EI) && error("No uncoupled beam-end alpha=0 branch was found in $sweep_path.")

    sample_total = min(n_sample, length(curve_EI))
    sample_idx = unique(round.(Int, range(1, length(curve_EI); length=sample_total)))
    sample_EI = curve_EI[sample_idx]
    sample_mp = curve_mp[sample_idx]

    println("Sampling $(length(sample_idx)) points along lowest uncoupled beam-end S~0 curve")

    all_q = zeros(ComplexF64, length(sample_idx), n_modes)
    all_energy = zeros(Float64, length(sample_idx), n_modes)
    all_phase = zeros(Float64, length(sample_idx), n_modes)
    mode_types = fill("", n_modes)

    for (ip, idx) in enumerate(sample_idx)
        EI_val = sample_EI[ip]
        mp_norm = sample_mp[ip]
        println("  case $ip / $(length(sample_idx)): EI = $(EI_val), x_M/L = $(round(mp_norm; digits=4))")
        params = FlexibleParams(;
            sigma=base.sigma,
            rho=base.rho,
            nu=base.nu,
            g=base.g,
            L_raft=base.L_raft,
            motor_position=mp_norm * base.L_raft,
            d=base.d,
            EI=EI_val,
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
        modal = decompose_raft_freefree_modes(result; num_modes=n_modes, verbose=false)

        all_q[ip, 1:length(modal.q)] .= modal.q
        all_energy[ip, 1:length(modal.energy_frac)] .= modal.energy_frac
        all_phase[ip, 1:length(modal.q)] .= rad2deg.(angle.(modal.q))
        if ip == 1
            mode_types[1:length(modal.mode_type)] .= modal.mode_type
        end
    end

    active_modes = min(6, n_modes)
    p1 = plot(title="Modal amplitudes", xlabel="log10(EI)", ylabel="|q_n|", yscale=:log10, legend=:best)
    p2 = plot(title="Modal energy distribution", xlabel="log10(EI)", ylabel="Energy fraction (%)", legend=:best)
    p3 = plot(title="Modal phases", xlabel="log10(EI)", ylabel="arg(q_n) [deg]", ylim=(-180, 180), legend=:best)
    p4 = plot(title="Complex q_n at EI=$(round(sample_EI[cld(length(sample_idx), 2)]; sigdigits=3))", xlabel="Re(q_n)", ylabel="Im(q_n)", legend=false, aspect_ratio=:equal)

    colors = Plots.palette(:auto, active_modes)
    for j in 1:active_modes
        label = "mode $(j - 1) ($(mode_types[j]))"
        color = colors[j]
        plot!(p1, log10.(sample_EI), abs.(all_q[:, j]); label=label, lw=2, marker=:circle, color=color)
        plot!(p2, log10.(sample_EI), 100 .* all_energy[:, j]; label=label, lw=2, marker=:circle, color=color)
        plot!(p3, log10.(sample_EI), all_phase[:, j]; label=label, lw=2, marker=:circle, color=color)
    end

    mid = cld(length(sample_idx), 2)
    for j in 1:active_modes
        scatter!(p4, [real(all_q[mid, j])], [imag(all_q[mid, j])]; markersize=6, color=colors[j])
        annotate!(p4, real(all_q[mid, j]), imag(all_q[mid, j]), text("q_$(j - 1)", 9, colors[j]))
    end

    fig = plot(p1, p2, p3, p4; layout=(2, 2), size=(1200, 900), plot_title="Modal decomposition along lowest uncoupled beam-end S≈0 curve")
    pdf_path = joinpath(data_dir, pdf_file)
    savefig(fig, pdf_path)

    csv_path = joinpath(data_dir, csv_file)
    write_modal_summary_csv(csv_path, sample_EI, sample_mp, mode_types, all_q, all_energy, all_phase)

    println("Saved $pdf_path")
    println("Saved $csv_path")
    return (pdf=pdf_path, csv=csv_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data")
    main(data_dir)
end
