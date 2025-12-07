module DtN

using LinearAlgebra

export DtN_generator

"""
    DtN_generator(N::Int, h::Float64 = 1.0)

This function generates the matrix M so that M * phi is an approximation of
1/π * lim_{ε→0} ∫_{|x-x0| > ε} (phi(x0, 0) - phi(x, 0)) / (x-x0)^2 dx

For harmonic functions in the plane with decaying behaviour, this is exactly
d/dz phi(x, 0)|_{x = x0}
"""
function DtN_generator(N::Int, h::Float64 = 1.0)
    # Create the main diagonal with 66's
    dtn = diagm(0 => fill(66.0, N))
    
    # Fill the first sub- and super-diagonals with -32's
    if N > 1
        dtn += diagm(1 => fill(-32.0, N-1))
        dtn += diagm(-1 => fill(-32.0, N-1))
    end
        
    # Fill the second sub- and super-diagonals with -1's
    if N > 2
        dtn += diagm(2 => fill(-1.0, N-2))
        dtn += diagm(-2 => fill(-1.0, N-2))
    end
        
    dtn = dtn / 18.0 # This is the integral around the origin
    dtn = dtn + diagm(0 => fill(1.0, N)) # First integral away of the origin. 
    
    # Now second integral away from the origin
    coefficients = zeros(Float64, N+1)
    
    coef(n, d) = -Float64(n)/(n+d) + (2*n - d)/2 * log((n+1)/(n-1)) - 1
    
    for jj in 1:div(N, 2)
        n = 2 * jj + 1
        
        if n > 0 
             coefficients[n] += coef(n, -1.0)
        end
        
        if n+2 <= N+1
            coefficients[n+2] += coef(n, +1.0)
        end

        if n+1 <= N+1
            coefficients[n+1] += -2*coef(n, 0.0)
        end
    end

    dtn3 = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            idx = abs((i-1) - (j-1)) # 0-based index difference
            dtn3[i, j] = coefficients[idx + 1]
        end
    end
    
    dtn = dtn + dtn3

    return h * dtn / pi
end

end # module
