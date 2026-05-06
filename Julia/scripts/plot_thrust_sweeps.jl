"""
plot_thrust_sweeps.jl

Three stacked panels with the same y-axis (Force/Length, N/m) and the same two curves:
  • Thrust  T/d           — full solver
  • S_xx                  — Longuet-Higgins proxy:
      S_xx = (ρg/4 + 3σk²/4)·(|η_L_domain|² − |η_R_domain|²)

Panels differ only in the x-axis:
  1. Motor-position sweep   x = xM / L           (fixed EI, ν = ν_water)
  2. Stiffness sweep        x = log₁₀ κ           (fixed xM, ν = ν_water;  κ = EI/(ρ_R L⁴ ω²))
  3. Reynolds sweep         x = log₁₀ Re          (fixed xM, EI;  Re = ω L²/ν,  ν ∈ [ν/100, 100ν])

A star marks the surferbot operating point on every panel.
All sweeps use the coupled base parameters with ν_water = 1e-6 m²/s.

Output: output/figures/thrust_sweeps.{pdf,png}  (single stacked figure)
Results cached to output/jld2/thrust_sweeps.jld2.
Re-running loads the cache and plots instantly.

Usage:
  julia --project=. scripts/plot_thrust_sweeps.jl
"""

using Surferbot
using JLD2
using Plots
using LaTeXStrings
using Printf

const CACHE_PATH   = joinpath(@__DIR__, "..", "output", "jld2", "thrust_sweeps.jld2")
const FIG_DIR      = joinpath(@__DIR__, "..", "output", "figures")
const N_SWEEP      = 20
const NU_WATER     = 1e-6

# ─── Per-solve extraction ─────────────────────────────────────────────────────
function compute_Sxx(result)
    args = result.metadata.args
    k    = Float64(real(args.k))
    pref = Float64(args.rho) * Float64(args.g) / 4 +
           3/4 * Float64(args.sigma) * k^2
    m    = Surferbot.Analysis.beam_edge_metrics(result)
    return pref * (abs2(m.eta_left_domain) - abs2(m.eta_right_domain))
end

function solve_one(bp_overrides, bp)
    p   = Surferbot.Sweep.apply_parameter_overrides(bp, bp_overrides)
    res = Surferbot.flexible_solver(p)
    d   = Float64(res.metadata.args.d)
    return res.thrust / d, compute_Sxx(res)
end

# ─── Three sweeps ─────────────────────────────────────────────────────────────
function run_sweep_xM(bp)
    L   = Float64(bp.L_raft)
    xs  = collect(range(0.0, 0.48; length = N_SWEEP))
    T   = Vector{Float64}(undef, N_SWEEP)
    Sxx = Vector{Float64}(undef, N_SWEEP)
    println("Sweep 1/3: motor position ($N_SWEEP points) …")
    for (i, xM_norm) in enumerate(xs)
        T[i], Sxx[i] = solve_one((motor_position = xM_norm * L, nu = 0.0), bp)
        @printf "  [%2d/%d]  xM/L=%.3f   T/d=%+.3e   Sxx=%+.3e\n" i N_SWEEP xM_norm T[i] Sxx[i]
    end
    return (; x = xs, thrust = T, Sxx)
end

function run_sweep_kappa(bp)
    rho_R = Float64(bp.rho_raft)
    L     = Float64(bp.L_raft)
    omega = Float64(bp.omega)
    xM    = Float64(bp.motor_position)
    EI_scale = rho_R * L^4 * omega^2        # EI = κ × EI_scale

    log10_kappa = collect(range(-4.0, 1.0; length = N_SWEEP))
    T   = Vector{Float64}(undef, N_SWEEP)
    Sxx = Vector{Float64}(undef, N_SWEEP)
    println("Sweep 2/3: stiffness κ ($N_SWEEP points) …")
    for (i, lk) in enumerate(log10_kappa)
        EI_i = 10.0^lk * EI_scale
        T[i], Sxx[i] = solve_one((EI = EI_i, motor_position = xM, nu = 0.0), bp)
        @printf "  [%2d/%d]  log10(κ)=%.2f   T/d=%+.3e   Sxx=%+.3e\n" i N_SWEEP lk T[i] Sxx[i]
    end
    return (; x = log10_kappa, thrust = T, Sxx)
end

function run_sweep_Re(bp)
    L     = Float64(bp.L_raft)
    omega = Float64(bp.omega)
    xM    = Float64(bp.motor_position)
    EI    = Float64(bp.EI)

    # ν from ν_water/100 to 100·ν_water  →  Re = ω L²/ν
    log10_nu = collect(range(log10(NU_WATER / 100), log10(NU_WATER * 100); length = N_SWEEP))
    log10_Re = log10(omega * L^2) .- log10_nu   # Re = ωL²/ν

    T   = Vector{Float64}(undef, N_SWEEP)
    Sxx = Vector{Float64}(undef, N_SWEEP)
    println("Sweep 3/3: Reynolds ($N_SWEEP points) …")
    for (i, lnu) in enumerate(log10_nu)
        nu_i = 10.0^lnu
        T[i], Sxx[i] = solve_one((EI = EI, motor_position = xM, nu = nu_i), bp)
        Re_i = 10.0^log10_Re[i]
        @printf "  [%2d/%d]  log10(Re)=%.2f   T/d=%+.3e   Sxx=%+.3e\n" i N_SWEEP log10_Re[i] T[i] Sxx[i]
    end
    return (; x = log10_Re, thrust = T, Sxx)
end

# ─── Surferbot operating point (one solve, used by all three figures) ─────────
function surferbot_point(bp)
    T, Sxx = solve_one((nu = NU_WATER,), bp)
    rho_R  = Float64(bp.rho_raft)
    L      = Float64(bp.L_raft)
    omega  = Float64(bp.omega)
    EI     = Float64(bp.EI)
    kappa  = EI / (rho_R * L^4 * omega^2)
    Re     = omega * L^2 / NU_WATER
    xM_norm = Float64(bp.motor_position) / L
    return (; xM_norm, log10_kappa = log10(kappa), log10_Re = log10(Re), thrust = T, Sxx)
end

# ─── Cache ────────────────────────────────────────────────────────────────────
function load_or_compute(bp)
    if isfile(CACHE_PATH)
        println("Loading cache from $CACHE_PATH …")
        d = JLD2.load(CACHE_PATH)
        sw1 = (; x = d["xM_x"],    thrust = d["xM_T"],    Sxx = d["xM_Sxx"])
        sw2 = (; x = d["kap_x"],   thrust = d["kap_T"],   Sxx = d["kap_Sxx"])
        sw3 = (; x = d["re_x"],    thrust = d["re_T"],     Sxx = d["re_Sxx"])
        sp  = (; xM_norm      = d["sp_xM"],
                log10_kappa   = d["sp_lk"],
                log10_Re      = d["sp_lRe"],
                thrust        = d["sp_T"],
                Sxx           = d["sp_Sxx"])
        return sw1, sw2, sw3, sp
    end

    sw1 = run_sweep_xM(bp)
    sw2 = run_sweep_kappa(bp)
    sw3 = run_sweep_Re(bp)
    sp  = surferbot_point(bp)

    mkpath(dirname(CACHE_PATH))
    JLD2.save(CACHE_PATH,
        "xM_x",   sw1.x,   "xM_T",   sw1.thrust,  "xM_Sxx",  sw1.Sxx,
        "kap_x",  sw2.x,   "kap_T",  sw2.thrust,  "kap_Sxx", sw2.Sxx,
        "re_x",   sw3.x,   "re_T",   sw3.thrust,  "re_Sxx",  sw3.Sxx,
        "sp_xM",  sp.xM_norm, "sp_lk", sp.log10_kappa, "sp_lRe", sp.log10_Re,
        "sp_T",   sp.thrust,  "sp_Sxx", sp.Sxx)
    println("Saved cache → $CACHE_PATH")
    return sw1, sw2, sw3, sp
end

# ─── Shared plot style ────────────────────────────────────────────────────────
const PLOT_OPTS = (
    ylabel     = L"\mathrm{Force\ /\ Length\ (N\,m^{-1})}",
    legend     = :topright,
    background_color_legend = RGBA(1, 1, 1, 0.85),
    foreground_color_legend = :black,
    size       = (900, 225),
    dpi        = 220,
    margin     = 5Plots.mm,
    framestyle = :box,
    grid       = false,
    guidefontsize  = 14,
    tickfontsize   = 12,
    titlefontsize  = 13,
    legendfontsize = 12,
    fontfamily = "Computer Modern",
)

function make_panel(sw, xlabel_str, title_str, sp_x, sp_thrust, sp_Sxx)
    p = plot(sw.x, sw.thrust;
             label = "Thrust  (solver)",
             color = :royalblue, linewidth = 2.5,
             xlabel = xlabel_str, title = title_str,
             PLOT_OPTS...)

    plot!(p, sw.x, sw.Sxx;
          label = L"S_{xx}\ \mathrm{(LH)}",
          color = :crimson, linewidth = 2.5, linestyle = :dash)

    hline!(p, [0.0]; color = :black, linewidth = 0.8, linestyle = :dot, label = false)

    # Surferbot operating point: star on the thrust curve (y = sp_thrust)
    scatter!(p, [sp_x], [sp_thrust];
             marker = :star5, markersize = 14,
             color = RGB(0.95, 0.75, 0.05),
             markerstrokecolor = :black, markerstrokewidth = 1,
             label = "Surferbot")

    return p
end

# ─── Main ─────────────────────────────────────────────────────────────────────
function main()
    bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    sw1, sw2, sw3, sp = load_or_compute(bp)

    p1 = make_panel(sw1,
        L"x_M / L",
        "Thrust vs Motor Position (coupled surferbot)",
        sp.xM_norm, sp.thrust, sp.Sxx)

    p2 = make_panel(sw2,
        L"\log_{10}\,\kappa",
        "Thrust vs Stiffness (coupled surferbot)",
        sp.log10_kappa, sp.thrust, sp.Sxx)

    p3 = make_panel(sw3,
        L"\log_{10}\,Re",
        "Thrust vs Reynolds Number (coupled surferbot)",
        sp.log10_Re, sp.thrust, sp.Sxx)

    fig = plot(p1, p2, p3; layout = (3, 1), size = (900, 675))
    mkpath(FIG_DIR)
    out_base = joinpath(FIG_DIR, "thrust_sweeps")
    savefig(fig, out_base * ".pdf")
    savefig(fig, out_base * ".png")
    println("Saved $(out_base).{pdf,png}")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
