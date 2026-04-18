using Surferbot
using LinearAlgebra
using DelimitedFiles

# Load the artifact to get base params and grid
data_dir = normpath(joinpath(@__DIR__, "..", "output"))
sweep_file = "sweep_motor_position_EI_uncoupled_from_matlab.jld2"
artifact = load_sweep(joinpath(data_dir, sweep_file))

# Load the branch data
branch_csv = "single_alpha_zero_curve_details_uncoupled_refined.csv"
lines = readlines(joinpath(data_dir, branch_csv))
header = split(lines[1], ",")

function get_col(header, name)
    findall(x -> x == name, header)
end

# Find columns for q0, q2, q4, q6
iq0 = findfirst(x -> x == "q0_re", header)
iq2 = findfirst(x -> x == "q2_re", header)
iq4 = findfirst(x -> x == "q4_re", header)
iq6 = findfirst(x -> x == "q6_re", header)
iEI = findfirst(x -> x == "EI", header)

# We need the mode shapes at the ends.
# The code uses weighted_mgs on phi_raw.
# Let's recreate the basis for a typical point.
# Grid size N depends on the case, but let's just use one from the artifact.
params = artifact.base_params
# Let's pick a high EI point from the branch.
rows = [split(l, ",") for l in lines[2:end]]
sort!(rows, by = r -> parse(Float64, r[iEI]), rev=true)

# Top 20
top_20 = rows[1:20]

# To get the exact Psi, we need to run a solver or at least setup the grid.
# But Psi is just the orthonormalized phi_raw.
# phi_raw depends on x_raft.
# Let's just run one flexible_solver to get the actual result object.
r1 = flexible_solver(apply_parameter_overrides(params, (EI=parse(Float64, top_20[1][iEI]), motor_position=parse(Float64, top_20[1][6]))))
modal = decompose_raft_freefree_modes(r1; num_modes=8, verbose=false)

# Now modal.Psi has the basis.
# modal.Psi is (Nr, n_modes).
# modal.Psi[1, :] are the values at the left end.
W = modal.Psi[1, :]
W_right = modal.Psi[end, :]

println("Mode labels: ", modal.n)
println("W_left: ", W)
println("W_right: ", W_right)

# Now check kappa for top 20
k02s = Float64[]
high_rels = Float64[]
k_fulls = Float64[]

for r in top_20
    # q is stored in the CSV but we can also just re-run or parse.
    # Parsing is easier.
    q0 = Complex(parse(Float64, r[iq0]), parse(Float64, r[iq0+1]))
    q2 = Complex(parse(Float64, r[iq2]), parse(Float64, r[iq2+1]))
    q4 = Complex(parse(Float64, r[iq4]), parse(Float64, r[iq4+1]))
    q6 = Complex(parse(Float64, r[iq6]), parse(Float64, r[iq6+1]))
    
    C0 = q0 * W[1]
    C2 = q2 * W[3] # index 3 because modal.n is [0, 1, 2, ...]
    C4 = q4 * W[5]
    C6 = q6 * W[7]
    
    kappa_02 = abs(C0 + C2) / (abs(C0) + abs(C2))
    higher_rel = (abs(C4) + abs(C6)) / (abs(C0) + abs(C2))
    kappa_full = abs(C0 + C2 + C4 + C6) / (abs(C0) + abs(C2) + abs(C4) + abs(C6))
    
    push!(k02s, kappa_02)
    push!(high_rels, higher_rel)
    push!(k_fulls, kappa_full)
end

using Statistics
println("kappa_02: median=", median(k02s), " max=", maximum(k02s))
println("higher_rel: median=", median(high_rels), " max=", maximum(high_rels))
println("kappa_full: median=", median(k_fulls), " max=", maximum(k_fulls))
