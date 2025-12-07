module Utils

using LinearAlgebra

export solve_tensor_system, gaussian_load, dispersion_k

"""
    solve_tensor_system(A, b; tol=1e-5, maxiter=1000)

Solve A·x = b.
Works for dense or sparse A, square or rectangular.
In Julia, `\\` handles both square (solve) and rectangular (least squares) cases efficiently.
"""
function solve_tensor_system(A, b; tol=1e-5, maxiter=1000)
    # Check dimensions
    # A is (m, n), b is (m,) or (m, k)
    if size(A, 1) != size(b, 1)
        throw(ArgumentError("b.shape must equal A.shape[0]"))
    end

    # Julia's backslash operator automatically handles:
    # - Square matrices: LU factorization (or similar)
    # - Rectangular matrices: QR factorization (least squares)
    # - Sparse matrices: Specialized solvers
    
    x = A \ b
    
    return x
end

"""
    gaussian_load(x0, sigma, x)

Smooth point load on an *arbitrary* 1-D grid.
"""
function gaussian_load(x0::Float64, sigma::Float64, x::AbstractVector{Float64})
    dx = diff(x)                              # length N-1
    
    w_mid = 0.5 * (dx[1:end-1] + dx[2:end])
    
    w = vcat(0.5 * dx[1], w_mid, 0.5 * dx[end])
    
    # Gaussian envelope
    phi = exp.(-0.5 * ((x .- x0) / sigma).^2)
    
    # exact discrete normalisation on this non-uniform grid
    delta = phi / sum(phi .* w)      # Σ delta_i w_i = 1
    return delta                        # q_i  (units N·m⁻¹)
end

"""
    dispersion_k(omega, g, H, nu, sigma, rho; k0=1.0 + 0.0im, num_steps=500)

Newton iteration for complex k using fixed number of steps.
"""
function dispersion_k(omega, g, H, nu, sigma, rho; k0=1.0 + 0.0im, num_steps=500)
    
    function dispersion_eq(k)
        tanh_kH = tanh(k * H)
        lhs = k * tanh_kH * g
        rhs = (-sigma / rho) * k^3 * tanh_kH + omega^2 - 4im * nu * omega * k^2
        return lhs - rhs
    end

    # Manual derivative for Newton's method
    # f(k) = g * k * tanh(kH) + (sigma/rho) * k^3 * tanh(kH) - omega^2 + 4i * nu * omega * k^2
    # f'(k) = g * (tanh(kH) + k * H * sech^2(kH)) + 
    #         (sigma/rho) * (3k^2 * tanh(kH) + k^3 * H * sech^2(kH)) + 
    #         8i * nu * omega * k
    
    function dispersion_deriv(k)
        tanh_kH = tanh(k * H)
        sech_kH_sq = sech(k * H)^2
        
        term1 = g * (tanh_kH + k * H * sech_kH_sq)
        term2 = (sigma / rho) * (3 * k^2 * tanh_kH + k^3 * H * sech_kH_sq)
        term3 = 8im * nu * omega * k
        
        return term1 + term2 + term3
    end

    k = k0
    for _ in 1:num_steps
        f = dispersion_eq(k)
        df_dk = dispersion_deriv(k)
        k = k - f / df_dk
    end
    
    return k
end

end # module
