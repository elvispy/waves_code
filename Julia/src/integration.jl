module Integration

export simpson_weights

"""
    simpson_weights(N::Int, h::Union{Float64, AbstractVector{Float64}})

Compute Simpson's 1/3 rule weights for numerical integration.

Supports both uniform and non-uniform grids.

# Arguments
- `N`: Number of grid points. Must be odd and ≥ 3. (Ignored if `h` is an array).
- `h`: Spacing parameter. 
    - If `Float64`: Uniform spacing between points.
    - If `AbstractVector`: Grid coordinates (must be sorted).

# Returns
- `w`: Vector of weights such that `sum(w .* f)` approximates the integral of `f`.
"""
function simpson_weights(N::Int, h::Union{Float64, AbstractVector{Float64}})
    # Non-uniform grid branch
    if isa(h, AbstractVector)
        x = h
        N = length(x)
        if N < 3 || N % 2 == 0
            throw(ArgumentError("Grid length must be odd and >= 3 for Simpson’s rule."))
        end
        w = zeros(eltype(x), N)

        # Composite Simpson: loop over each pair of intervals [x[i], x[i+1], x[i+2]]
        for i in 1:2:(N - 2)
            x0, x1, x2 = x[i], x[i+1], x[i+2]
            h0 = x1 - x0
            h1 = x2 - x1

            # Exact quadratic-interpolation weights:
            w0 =   h0/3 +  h1/6   - (h1^2)/(6*h0)
            w1 =  (h0^2)/(6*h1) + h0/2 + h1/2 + (h1^2)/(6*h0)
            w2 =  -(h0^2)/(6*h1) + h0/6 + h1/3

            w[i]   += w0
            w[i+1] += w1
            w[i+2] += w2
        end

        return w

    # Uniform grid branch
    else
        if N < 3 || N % 2 == 0
            throw(ArgumentError("N must be an odd integer >= 3 for Simpson’s rule."))
        end
        weights = ones(Float64, N)
        weights[2:2:N-1] .= 4.0
        weights[3:2:N-2] .= 2.0
        
        return weights * (h / 3.0)
    end
end

end # module
