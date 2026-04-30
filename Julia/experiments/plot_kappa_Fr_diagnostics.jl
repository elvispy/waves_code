using CSV
using DataFrames
using Plots
using LaTeXStrings
using Printf

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

# ─── Render one panel with the same publication style as plot_dimensionless_*
function render_panel(grid, fig_title, out_base; xlims=nothing, ylims=nothing)
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

        out_base = joinpath(fig_dir, "plot_kappa_Fr_$(case.label)")
        render_panel(grid, fig_title, out_base)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
