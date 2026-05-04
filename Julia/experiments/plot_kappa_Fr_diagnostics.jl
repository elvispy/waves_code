using CSV
using DataFrames
using LinearAlgebra
using Plots
using LaTeXStrings
using Printf
using Surferbot

include(joinpath(@__DIR__, "prescribed_wn_diagonal_impedance.jl"))
const ModalPressureMap = Main.PrescribedWnDiagonalImpedance

"""
plot_kappa_Fr_diagnostics.jl

α heatmap on the (log10 κ, log10 Fr) plane, one panel per coupling case.
Reads the CSVs produced by `sweeper_kappa_Fr.jl`. No scatter overlay yet —
just the underlying field.

Styling follows `plot_dimensionless_diagnostics.jl` so the two figures sit
side-by-side cleanly.
"""

# ─── Load α(log10 κ, log10 Fr) from a CSV produced by sweeper_kappa_Fr.jl ────
function load_alpha_grid(csv_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)
    log10_kappa = sort(unique(df.log10_kappa))
    log10_Fr    = sort(unique(df.log10_Fr))
    alpha = fill(NaN, length(log10_Fr), length(log10_kappa))
    lookup = Dict{Tuple{Float64,Float64}, Float64}(
        (row.log10_kappa, row.log10_Fr) => row.alpha for row in eachrow(df))
    for (j, lk) in enumerate(log10_kappa), (i, lFr) in enumerate(log10_Fr)
        if haskey(lookup, (lk, lFr))
            alpha[i, j] = lookup[(lk, lFr)]
        end
    end
    return (; log10_kappa, log10_Fr, alpha,
              d        = first(df.d),
              L_raft   = first(df.L_raft),
              xM_over_L = first(df.xM_over_L))
end

# ─── Canonical SurferBot point in (log10 κ, log10 Fr) ───────────────────────
function surferbot_kappa_Fr(is_coupled::Bool)
    bp = is_coupled ?
        Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params :
        Surferbot.Analysis.default_uncoupled_motor_position_EI_sweep().base_params
    kappa = bp.EI / (bp.rho_raft * bp.L_raft^4 * bp.omega^2)
    Fr    = sqrt(bp.L_raft * bp.omega^2 / bp.g)
    return log10(kappa), log10(Fr)
end

# ─── Resonance κ prediction via generalized eigenvalue problem ───────────────
#
# For fixed Fr, D_m(κ) = κ·b_m + D0_m is linear in κ.  The system matrix
# A(κ) = Diagonal(D(κ)) − Z_ψ is singular when det A = 0, which is equivalent
# to the standard eigenvalue problem C·v = κ·v with
#
#   C = diag(b)⁻¹ · (Z_ψ − diag(D₀))
#
# One N×N eigenvalue problem per Fr value; eigenvalues are generally complex.
# We scatter Re(κ) > 0 on the (log₁₀κ, log₁₀Fr) heatmap.

function kappa_resonances_at_Fr(base_params, Fr::Real; output_dir, num_modes=8)
    L     = base_params.L_raft
    omega = base_params.omega
    rho_R = base_params.rho_raft isa AbstractVector ?
            minimum(base_params.rho_raft) : float(base_params.rho_raft)

    g_eff     = L * omega^2 / Fr^2
    params_Fr = Surferbot.Sweep.apply_parameter_overrides(base_params, (g = g_eff,))

    payload = ModalPressureMap.load_or_compute_modal_pressure_map(
                  params_Fr; output_dir = output_dir, num_modes_basis = num_modes)
    Z_psi   = ComplexF64.(payload.Z_psi)
    beta    = collect(Float64.(payload.beta))   # β_m, dimensional (1/m)
    d_eff   = Float64(Surferbot.derive_params(params_Fr).d)

    # κ-slope of D_m:  b_m = ρ_R L^4 ω^2 β_m^4
    b  = rho_R .* L^4 .* omega^2 .* beta .^ 4
    # κ-independent part of D_m:  D0_m = −ρ_R ω^2 + d ρ g
    D0 = fill(-rho_R * omega^2 + base_params.rho * g_eff * d_eff, length(beta))

    C = Diagonal(1.0 ./ b) * (Z_psi - Diagonal(ComplexF64.(D0)))
    return eigvals(Matrix(C))
end

function overlay_kappa_resonances!(plt, base_params, log10_Fr_vals;
                                    output_dir, num_modes=8)
    xs = Float64[]; ys = Float64[]
    for log10_Fr in log10_Fr_vals
        Fr = 10.0^log10_Fr
        kappa_eigs = kappa_resonances_at_Fr(base_params, Fr;
                                             output_dir = output_dir,
                                             num_modes  = num_modes)
        for κ in kappa_eigs
            Re_kappa = real(κ)
            Re_kappa > 0 || continue
            push!(xs, log10(Re_kappa))
            push!(ys, log10_Fr)
        end
        println("  resonance κ done for log10(Fr) = $(round(log10_Fr; digits=3))")
    end
    scatter!(plt, xs, ys;
             marker           = :circle,
             markersize        = 6,
             color             = :white,
             markerstrokecolor = :black,
             markerstrokewidth = 1.2,
             label             = L"\det(D - Z_\psi) = 0")
end

# ─── Render one panel with the same publication style as plot_dimensionless_*
function render_panel(grid, fig_title, out_base, is_coupled;
                      xlims=nothing, ylims=nothing,
                      base_params=nothing, output_dir=nothing, n_eig_Fr=15)
    XLIMS = something(xlims, (minimum(grid.log10_kappa), maximum(grid.log10_kappa)))
    YLIMS = something(ylims, (minimum(grid.log10_Fr),    maximum(grid.log10_Fr)))

    plt_opts = (
        xlabel = L"\log_{10}\,\kappa",
        ylabel = L"\log_{10}\,Fr",
        colormap = :balance,
        clims = (-1, 1),
        levels = 51,
        interpolate = true,
        xlims = XLIMS,
        ylims = YLIMS,
        legend = false,
        size = (820, 640),
        margin = 6Plots.mm,
        dpi = 220,
        titlefontsize = 14,
        guidefontsize = 14,
        tickfontsize = 12,
        fontfamily = "Computer Modern",
        framestyle = :box,
        grid = false,
        colorbar_title = L"\alpha",
        colorbar_titlefontsize = 14,
        colorbar_tickfontsize = 11,
    )

    p = heatmap(grid.log10_kappa, grid.log10_Fr, grid.alpha;
                title = fig_title, plt_opts...)

    lk_star, lFr_star = surferbot_kappa_Fr(is_coupled)
    scatter!(p, [lk_star], [lFr_star];
             marker = :star5, markersize = 12,
             color = :white, markerstrokecolor = :black, markerstrokewidth = 1,
             label = false)

    if !isnothing(base_params) && !isnothing(output_dir)
        log10_Fr_vals = range(YLIMS[1], YLIMS[2]; length = n_eig_Fr)
        overlay_kappa_resonances!(p, base_params, collect(log10_Fr_vals);
                                   output_dir = output_dir)
    end

    savefig(p, out_base * ".pdf")
    savefig(p, out_base * ".png")
    println("Saved $(out_base).{pdf,png}")
end

# ─── Linear (κ, 1/Fr²) panel — to check whether α=0 contours are straight ───
# Uses scatter so every data point sits at its exact (κ, 1/Fr²) position.
function render_linear_panel(grid, fig_title, out_base, is_coupled)
    # Flatten grid into parallel vectors
    nFr, nk = length(grid.log10_Fr), length(grid.log10_kappa)
    kappa_vec   = vec([10^grid.log10_kappa[j]      for _ in 1:nFr, j in 1:nk])
    inv_Fr2_vec = vec([10^(-2*grid.log10_Fr[i])    for i in 1:nFr, _ in 1:nk])
    alpha_vec   = vec(grid.alpha)

    lk_star, lFr_star = surferbot_kappa_Fr(is_coupled)
    kappa_star  = 10^lk_star
    invFr2_star = 10^(-2 * lFr_star)

    p = scatter(kappa_vec, inv_Fr2_vec;
                marker_z = alpha_vec,
                colormap = :balance,
                clims = (-1, 1),
                markersize = 3,
                markerstrokewidth = 0,
                xlabel = L"\kappa",
                ylabel = L"Fr^{-2}",
                title = fig_title,
                legend = false,
                size = (820, 640),
                margin = 6Plots.mm,
                dpi = 220,
                titlefontsize = 14,
                guidefontsize = 14,
                tickfontsize = 12,
                fontfamily = "Computer Modern",
                framestyle = :box,
                grid = false,
                colorbar_title = L"\alpha",
                colorbar_titlefontsize = 14,
                colorbar_tickfontsize = 11,
    )
    scatter!(p, [kappa_star], [invFr2_star];
             marker = :star5, markersize = 12,
             color = :white, markerstrokecolor = :black, markerstrokewidth = 1,
             label = false)

    savefig(p, out_base * ".pdf")
    savefig(p, out_base * ".png")
    println("Saved $(out_base).{pdf,png}")
end

# ─── Main ────────────────────────────────────────────────────────────────────
function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    csv_dir    = joinpath(output_dir, "csv")
    fig_dir    = joinpath(output_dir, "figures")
    mkpath(fig_dir)

    cases = [
        (label = "uncoupled", csv = "sweeper_kappa_Fr_uncoupled.csv"),
        (label = "coupled",   csv = "sweeper_kappa_Fr_coupled.csv"),
    ]

    for case in cases
        csv_path = joinpath(csv_dir, case.csv)
        if !isfile(csv_path)
            @warn "$(case.csv) not found in $(csv_dir) — run sweeper_kappa_Fr.jl first; skipping."
            continue
        end

        println("Processing $(case.label) ...")
        grid = load_alpha_grid(csv_path)

        # Λ = d/L (0 for uncoupled, d/L for coupled). Echoed from the CSV row.
        adj        = case.label == "coupled" ? "Coupled" : "Uncoupled"
        Lambda_val = grid.d == 0 ? "0" : @sprintf("%.2f", grid.d / grid.L_raft)
        xM_str     = @sprintf("%.2f", grid.xM_over_L)
        fig_title  = LaTeXString(
            "$adj raft, \$\\Lambda = $Lambda_val\$, \$x_M/L = $xM_str\$ — \$\\alpha\$ field")

        is_coupled = case.label == "coupled"
        bp         = is_coupled ?
            Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params :
            Surferbot.Analysis.default_uncoupled_motor_position_EI_sweep().base_params
        out_base   = joinpath(fig_dir, "plot_kappa_Fr_$(case.label)")
        render_panel(grid, fig_title, out_base, is_coupled;
                     base_params = bp, output_dir = output_dir)

        if !is_coupled
            render_linear_panel(grid, fig_title, out_base * "_linear", is_coupled)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
