using Surferbot
using DelimitedFiles
using Statistics

"""
check_uncoupled_energy_distribution.jl

Analyzes the fractional modal energy distribution (symmetric vs. antisymmetric) 
along the highest stiffness alpha=0 branch points.
"""

data_dir = normpath(joinpath(@__DIR__, "..", "output"))
branch_csv = joinpath("csv", "analyze_single_alpha_zero_curve.csv")
lines = readlines(joinpath(data_dir, branch_csv))
header = split(lines[1], ",")

function get_col_idx(name)
    findfirst(x -> x == name, header)
end

iEI = get_col_idx("EI")
iq0 = get_col_idx("q0_re")
iq2 = get_col_idx("q2_re")
iq4 = get_col_idx("q4_re")
iq6 = get_col_idx("q6_re")
ie0 = get_col_idx("energy_frac0")
ie2 = get_col_idx("energy_frac2")
ie4 = get_col_idx("energy_frac4")
ie6 = get_col_idx("energy_frac6")

rows = [split(l, ",") for l in lines[2:end]]
sort!(rows, by = r -> parse(Float64, r[iEI]), rev=true)

top_20 = rows[1:20]

# We need the basis functions.
# Just run one solver for the highest EI point.
EI_max = parse(Float64, top_20[1][iEI])
xM_max = parse(Float64, top_20[1][get_col_idx("motor_position")])

sweep_file = joinpath("jld2", "sweep_motor_position_EI_uncoupled_from_matlab.jld2")
artifact = load_sweep(joinpath(data_dir, sweep_file))
params = apply_parameter_overrides(artifact.base_params, (EI=EI_max, motor_position=xM_max))
result = flexible_solver(params)
modal = decompose_raft_freefree_modes(result; num_modes=8, verbose=false)

W = modal.Psi[1, :] # Left end
W_right = modal.Psi[end, :]

println("W_left (modes 0,2,4,6): ", W[1], ", ", W[3], ", ", W[5], ", ", W[7])

k02s = Float64[]
high_rels = Float64[]
k_fulls = Float64[]
sym_energy_02_fracs = Float64[]

for r in top_20
    q0 = Complex(parse(Float64, r[iq0]), parse(Float64, r[iq0+1]))
    q2 = Complex(parse(Float64, r[iq2]), parse(Float64, r[iq2+1]))
    q4 = Complex(parse(Float64, r[iq4]), parse(Float64, r[iq4+1]))
    q6 = Complex(parse(Float64, r[iq6]), parse(Float64, r[iq6+1]))
    
    C0 = q0 * W[1]
    C2 = q2 * W[3]
    C4 = q4 * W[5]
    C6 = q6 * W[7]
    
    kappa_02 = abs(C0 + C2) / (abs(C0) + abs(C2))
    higher_rel = (abs(C4) + abs(C6)) / (abs(C0) + abs(C2))
    kappa_full = abs(C0 + C2 + C4 + C6) / (abs(C0) + abs(C2) + abs(C4) + abs(C6))
    
    push!(k02s, kappa_02)
    push!(high_rels, higher_rel)
    push!(k_fulls, kappa_full)
    
    e0 = parse(Float64, r[ie0])
    e2 = parse(Float64, r[ie2])
    e4 = parse(Float64, r[ie4])
    e6 = parse(Float64, r[ie6])
    
    sym_total = e0 + e2 + e4 + e6
    push!(sym_energy_02_fracs, (e0 + e2) / sym_total)
end

println("kappa_02: median=", median(k02s), " max=", maximum(k02s))
println("higher_rel: median=", median(high_rels), " max=", maximum(high_rels))
println("kappa_full: median=", median(k_fulls), " max=", maximum(k_fulls))
println("Symmetric energy in 0+2 frac: median=", median(sym_energy_02_fracs))
println("EI range: ", parse(Float64, top_20[20][iEI]), " to ", EI_max)
