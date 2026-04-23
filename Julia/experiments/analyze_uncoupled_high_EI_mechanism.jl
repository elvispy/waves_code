using Surferbot
using Base.Threads
using DelimitedFiles

# Purpose: run local x_M cuts around the top-20 highest-EI points on the
# uncoupled beam-based alpha=0 branch and classify the local mechanism.

function wrap_to_pi(x)
    y = mod(x + pi, 2pi) - pi
    return y == -pi ? pi : y
end

function parse_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("Empty CSV: $path")
    header = split(lines[1], ",")
    rows = Vector{Dict{String, String}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ",")
        length(vals) == length(header) || continue
        push!(rows, Dict(header[i] => vals[i] for i in eachindex(header)))
    end
    return rows
end

function top_high_EI_rows(rows; top_n::Int=20)
    sorted = sort(rows; by = r -> parse(Float64, r["EI"]), rev=true)
    return sorted[1:min(top_n, length(sorted))]
end

function build_point_summary(result)
    metrics = beam_edge_metrics(result)
    S = (metrics.eta_right_beam + metrics.eta_left_beam) / 2
    A = (metrics.eta_right_beam - metrics.eta_left_beam) / 2
    Psi = real(S * conj(A))
    phase_gap = wrap_to_pi(angle(S) - angle(A))
    modal = decompose_raft_freefree_modes(result; num_modes=8, verbose=false)
    sym_energy = sum(modal.energy_frac[i] for i in eachindex(modal.energy_frac) if iseven(modal.n[i]))
    antisym_energy = sum(modal.energy_frac[i] for i in eachindex(modal.energy_frac) if isodd(modal.n[i]))
    return (
        eta_left_beam = metrics.eta_left_beam,
        eta_right_beam = metrics.eta_right_beam,
        alpha_beam = beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam),
        S = S,
        A = A,
        Psi = Psi,
        phase_gap = phase_gap,
        sym_energy = sym_energy,
        antisym_energy = antisym_energy,
        modal = modal,
    )
end

function classify_cut(points)
    idx = argmin(abs.(getfield.(points, :Psi)))
    p = points[idx]
    ratio = abs(p.S) / max(abs(p.A), eps())
    invratio = abs(p.A) / max(abs(p.S), eps())
    cosgap = p.Psi / max(abs(p.S) * abs(p.A), eps())
    if ratio < 0.25
        return ("S-zero", idx, ratio, cosgap)
    elseif invratio < 0.25
        return ("A-zero", idx, ratio, cosgap)
    elseif abs(cosgap) < 0.15
        return ("phase-orthogonal", idx, ratio, cosgap)
    else
        return ("mixed/interference", idx, ratio, cosgap)
    end
end

function run_case(base_params, EI, xM_over_L)
    params = apply_parameter_overrides(base_params, (EI = EI, motor_position = xM_over_L * base_params.L_raft))
    result = flexible_solver(params)
    return build_point_summary(result)
end

function write_raw_csv(path, rows)
    open(path, "w") do io
        println(io, "ei_rank,EI,xM_center_over_L,delta_xM_over_L,xM_over_L,alpha_beam,Psi,S_abs,A_abs,phase_gap_deg,sym_energy,antisym_energy")
        for r in rows
            println(io, join([
                string(r.ei_rank),
                string(r.EI),
                string(r.xM_center_over_L),
                string(r.delta_xM_over_L),
                string(r.xM_over_L),
                string(r.alpha_beam),
                string(r.Psi),
                string(r.S_abs),
                string(r.A_abs),
                string(rad2deg(r.phase_gap)),
                string(r.sym_energy),
                string(r.antisym_energy),
            ], ","))
        end
    end
end

function write_summary_csv(path, rows)
    open(path, "w") do io
        println(io, "ei_rank,EI,xM_center_over_L,alpha_center,mechanism,ratio_S_over_A,cos_phase_gap,sym_energy_center,antisym_energy_center")
        for r in rows
            println(io, join([
                string(r.ei_rank),
                string(r.EI),
                string(r.xM_center_over_L),
                string(r.alpha_center),
                r.mechanism,
                string(r.ratio_S_over_A),
                string(r.cos_phase_gap),
                string(r.sym_energy_center),
                string(r.antisym_energy_center),
            ], ","))
        end
    end
end

function main(; data_dir=joinpath(@__DIR__, "..", "output"),
                branch_csv=joinpath("csv", "analyze_single_alpha_zero_curve.csv"),
                sweep_file=joinpath("jld2", "sweep_motor_position_EI_uncoupled_from_matlab.jld2"),
                top_n::Int=20,
                offsets=[-0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015],
                raw_output="analyze_uncoupled_high_EI_mechanism_raw.csv",
                summary_output="analyze_uncoupled_high_EI_mechanism_summary.csv")
    rows = parse_csv(joinpath(data_dir, branch_csv))
    top_rows = top_high_EI_rows(rows; top_n=top_n)
    artifact = load_sweep(joinpath(data_dir, sweep_file))
    xmin = minimum(artifact.parameter_axes.motor_position) / artifact.base_params.L_raft
    xmax = maximum(artifact.parameter_axes.motor_position) / artifact.base_params.L_raft

    raw = Vector{NamedTuple}()
    summary = Vector{NamedTuple}(undef, length(top_rows))
    lock = ReentrantLock()

    @threads for i in eachindex(top_rows)
        row = top_rows[i]
        EI = parse(Float64, row["EI"])
        x0 = parse(Float64, row["xM_over_L"])
        pts = NamedTuple[]
        for δ in offsets
            x = clamp(x0 + δ, xmin, xmax)
            out = run_case(artifact.base_params, EI, x)
            push!(pts, (; xM_over_L=x, delta_xM_over_L=δ, out...))
        end
        mechanism, idx, ratio, cosgap = classify_cut(pts)
        center_idx = findfirst(p -> isapprox(p.delta_xM_over_L, 0.0; atol=1e-12), pts)
        center = pts[center_idx === nothing ? idx : center_idx]
        local_raw = [(
            ei_rank=i,
            EI=EI,
            xM_center_over_L=x0,
            delta_xM_over_L=p.delta_xM_over_L,
            xM_over_L=p.xM_over_L,
            alpha_beam=p.alpha_beam,
            Psi=p.Psi,
            S_abs=abs(p.S),
            A_abs=abs(p.A),
            phase_gap=p.phase_gap,
            sym_energy=p.sym_energy,
            antisym_energy=p.antisym_energy,
        ) for p in pts]
        summary[i] = (
            ei_rank=i,
            EI=EI,
            xM_center_over_L=x0,
            alpha_center=center.alpha_beam,
            mechanism=mechanism,
            ratio_S_over_A=ratio,
            cos_phase_gap=cosgap,
            sym_energy_center=center.sym_energy,
            antisym_energy_center=center.antisym_energy,
        )
        lock(lock) do
            append!(raw, local_raw)
            println("EI rank $(i)/$(length(top_rows)): EI=$(EI), x_M/L=$(round(x0; digits=4)), alpha_center=$(round(center.alpha_beam; sigdigits=5)), mechanism=$(mechanism)")
        end
    end

    raw = sort(raw; by = r -> (r.ei_rank, r.delta_xM_over_L))
    raw_path = joinpath(data_dir, "csv", raw_output)
    summary_path = joinpath(data_dir, "csv", summary_output)
    write_raw_csv(raw_path, raw)
    write_summary_csv(summary_path, summary)
    println("Saved $raw_path")
    println("Saved $summary_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
