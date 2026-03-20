module Utils

using LinearAlgebra

export solve_tensor_system, gaussian_load, dispersion_k

erf_libm(x::Float64) = ccall((:erf, Base.Math.libm), Float64, (Float64,), x)

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
    
    x = nothing
    try
        x = A \ b
    catch err
        if err isa LinearAlgebra.SingularException
            x = qr(Matrix(A)) \ b
        else
            rethrow(err)
        end
    end
    
    return x
end

"""
    gaussian_load(x0, sigma, x)

Smooth point load on an *arbitrary* 1-D grid.
"""
function gaussian_load(x0::Float64, sigma::Float64, x::AbstractVector{Float64})
    a = -0.5
    b = 0.5
    xcol = collect(x)
    phi = exp.(-0.5 .* ((xcol .- x0) ./ sigma).^2)
    Z = sigma * sqrt(pi / 2) * (erf_libm((b - x0) / (sqrt(2) * sigma)) - erf_libm((a - x0) / (sqrt(2) * sigma)))
    return reshape((1 / Z) .* phi, length(xcol))
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

    k = k0
    for _ in 1:num_steps
        f = dispersion_eq(k)
        dk = 1e-8
        f_plus = dispersion_eq(k + dk)
        f_minus = dispersion_eq(k - dk)
        df = (f_plus - f_minus) / (2 * dk)
        k = k - f / df
    end
    return k
end

end # module
