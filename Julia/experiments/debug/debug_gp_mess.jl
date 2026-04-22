
using JLD2
using Statistics
using LinearAlgebra
using Printf

# Include the GP logic from the main script to be consistent
function fit_gp2d(x::AbstractVector, y::AbstractVector, values::AbstractVector)
    n = length(values)
    mean_value = mean(values)
    centered = collect(Float64.(values .- mean_value))

    dx = diff(sort(unique(Float64.(x))))
    dy = diff(sort(unique(Float64.(y))))
    dx = dx[dx .> 0]
    dy = dy[dy .> 0]
    lx = isempty(dx) ? 0.05 : max(0.02, 4 * median(dx))
    ly = isempty(dy) ? 0.15 : max(0.05, 4 * median(dy))
    sigma_f = max(std(values), 1e-3)
    noise = max(1e-4, 2e-2 * sigma_f)

    K = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n, j in i:n
        r2 = ((Float64(x[i]) - Float64(x[j])) / lx)^2 + ((Float64(y[i]) - Float64(y[j])) / ly)^2
        kij = sigma_f^2 * exp(-0.5 * r2)
        K[i, j] = kij
        K[j, i] = kij
    end
    @inbounds for i in 1:n
        K[i, i] += noise^2 + 1e-10
    end

    F = cholesky(Symmetric(K))
    weights = F \ centered
    return (x = Float64.(x), y = Float64.(y), weights = weights, mean = mean_value, lx = lx, ly = ly, sigma_f2 = sigma_f^2, noise=noise)
end

function predict_gp2d(model, xq::Real, yq::Real)
    acc = 0.0
    @inbounds for i in eachindex(model.weights)
        r2 = ((xq - model.x[i]) / model.lx)^2 + ((yq - model.y[i]) / model.ly)^2
        acc += model.weights[i] * (model.sigma_f2 * exp(-0.5 * r2))
    end
    return model.mean + acc
end

function main()
    path = "Julia/output/sweep_motor_position_EI_coupled_from_matlab.jld2"
    artifact = load(path, "artifact")
    
    motor_position_list = vec(Float64.(artifact.parameter_axes.motor_position))
    EI_list = vec(Float64.(artifact.parameter_axes.EI))
    mp_norm_list = motor_position_list ./ artifact.base_params.L_raft
    logEI_list = log10.(EI_list)
    
    # Reconstruct the grid values for alpha_beam
    summaries = artifact.summaries
    alpha_grid = zeros(length(mp_norm_list), length(EI_list))
    for ie in eachindex(EI_list), im in eachindex(mp_norm_list)
        s = summaries[im, ie]
        # alpha_beam calculation
        left = s.eta_left_beam
        right = s.eta_right_beam
        alpha_grid[im, ie] = (abs(right) - abs(left)) / (abs(right) + abs(left) + 1e-10)
    end
    
    xtrain, ytrain, vtrain = Float64[], Float64[], Float64[]
    for ie in eachindex(logEI_list), im in eachindex(mp_norm_list)
        push!(xtrain, mp_norm_list[im])
        push!(ytrain, logEI_list[ie])
        push!(vtrain, alpha_grid[im, ie])
    end
    
    println("GP Training Stats:")
    println("  N points: ", length(vtrain))
    println("  xM range: [", minimum(xtrain), ", ", maximum(xtrain), "]")
    println("  logEI range: [", minimum(ytrain), ", ", maximum(ytrain), "]")
    
    model = fit_gp2d(xtrain, ytrain, vtrain)
    println("\nHyperparameters:")
    println("  lx (motor): ", model.lx)
    println("  ly (logEI): ", model.ly)
    println("  sigma_f:    ", sqrt(model.sigma_f2))
    println("  noise:      ", model.noise)
    
    # Check the predicted roots vs real roots at log10(EI) = -4.33
    target_logEI = -4.329952
    println("\nInspecting log10(EI) = ", target_logEI)
    
    mp_dense = collect(range(0.0, 0.5, length=501))
    preds = [predict_gp2d(model, mp, target_logEI) for mp in mp_dense]
    
    println("\nGP Root Proposals (where alpha crosses zero):")
    for i in 1:(length(mp_dense)-1)
        if preds[i] * preds[i+1] <= 0
            t = preds[i] / (preds[i] - preds[i+1])
            root = mp_dense[i] + t * (mp_dense[i+1] - mp_dense[i])
            @printf("  Root at xM/L = %.4f (val_left=%.4f, val_right=%.4f)\n", root, preds[i], preds[i+1])
        end
    end
end

main()
