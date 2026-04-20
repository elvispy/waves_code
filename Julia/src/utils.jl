module Utils

using LinearAlgebra
using SpecialFunctions
using SparseArrays
using ForwardDiff

export solve_tensor_system, gaussian_load, dispersion_k

erf_libm(x::Float64) = ccall((:erf, Base.Math.libm), Float64, (Float64,), x)

# Helpers to handle Dual numbers (potentially nested)
_extract_value(x::Real) = ForwardDiff.value(x)
_extract_value(x::Complex) = Complex(_extract_value(real(x)), _extract_value(imag(x)))
_extract_value(x::AbstractArray) = _extract_value.(x)

_extract_partials(x::ForwardDiff.Dual, p_idx) = p_idx <= ForwardDiff.npartials(x) ? ForwardDiff.partials(x)[p_idx] : 0.0
_extract_partials(x::Real, p_idx) = 0.0
_extract_partials(x::Complex, p_idx) = Complex(_extract_partials(real(x), p_idx), _extract_partials(imag(x), p_idx))
_extract_partials(x::AbstractArray, p_idx) = _extract_partials.(x, p_idx)

"""
    solve_tensor_system(A, b; tol=1e-5, maxiter=1000)

Solve A·x = b.
Works for dense or sparse A, square or rectangular.
Special handling for SparseMatrixCSC with Dual numbers to avoid UMFPACK StackOverflow.
"""
function solve_tensor_system(A, b; tol=1e-5, maxiter=1000)
    # Check dimensions
    if size(A, 1) != size(b, 1)
        throw(ArgumentError("b.shape must equal A.shape[0]"))
    end

    # Detect if we have Dual numbers in a SparseMatrix
    if A isa SparseMatrixCSC && (eltype(A) <: ForwardDiff.Dual || eltype(A) <: Complex{<:ForwardDiff.Dual} ||
                                eltype(b) <: ForwardDiff.Dual || eltype(b) <: Complex{<:ForwardDiff.Dual})
        return _solve_sparse_dual(A, b)
    end

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
    _solve_sparse_dual(A, b)

Internal helper to solve sparse linear systems with Dual numbers by 
separating values and partials.
"""
function _solve_sparse_dual(A::SparseMatrixCSC, b)
    A0 = _extract_value(A)
    b0 = _extract_value(b)
    
    # Pre-factorize for efficiency
    fact = lu(A0)
    x0 = fact \ b0
    
    # We need to know the number of parameters p
    first_dual = nothing
    if eltype(A) <: ForwardDiff.Dual
        first_dual = A.nzval[1]
    elseif eltype(A) <: Complex{<:ForwardDiff.Dual}
        first_dual = real(A.nzval[1])
    elseif eltype(b) <: ForwardDiff.Dual
        first_dual = b[1]
    elseif eltype(b) <: Complex{<:ForwardDiff.Dual}
        first_dual = real(b[1])
    end
    
    if isnothing(first_dual)
        return x0
    end
    
    p = ForwardDiff.npartials(first_dual)
    T_dual_real = typeof(first_dual)
    T_val = eltype(x0)
    
    # Reconstruct partials of x
    x_part_raw = zeros(T_val, length(x0), p)
    for p_idx in 1:p
        # Extract partials of A and b for this parameter
        Ap_nzval = _extract_partials(A.nzval, p_idx)
        Ap = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, Ap_nzval)
        bp = _extract_partials(b, p_idx)
        
        # Solve A0 * x1 = bp - Ap * x0
        rhs_p = bp - Ap * x0
        x_part_raw[:, p_idx] = fact \ rhs_p
    end
    
    # Reconstruct the Dual vector result
    x = map(1:length(x0)) do i
        re_dual = T_dual_real(real(x0[i]), ForwardDiff.Partials(Tuple(real(x_part_raw[i, :]))))
        if x0 isa Vector{<:Complex}
             im_dual = T_dual_real(imag(x0[i]), ForwardDiff.Partials(Tuple(imag(x_part_raw[i, :]))))
             return Complex(re_dual, im_dual)
        else
             return re_dual
        end
    end
    
    return x
end

"""
    gaussian_load(x0, sigma, x)

Smooth point load on an *arbitrary* 1-D grid.
"""
function gaussian_load(x0::T, sigma::T, x::AbstractVector{T}) where {T<:Real}
    a = T(-0.5)
    b = T(0.5)
    xcol = collect(x)
    phi = exp.(-T(0.5) .* ((xcol .- x0) ./ sigma).^2)
    Z = sigma * sqrt(T(pi) / 2) * (erf((b - x0) / (sqrt(T(2)) * sigma)) - erf((a - x0) / (sqrt(T(2)) * sigma)))
    return reshape((1 / Z) .* phi, length(xcol))
end

"""
    dispersion_k(omega, g, H, nu, sigma, rho; k0=1.0 + 0.0im, num_steps=500)

Newton iteration for complex k using fixed number of steps.
"""
function dispersion_k(omega::T, g::T, H::T, nu::T, sigma::T, rho::T; k0=Complex{T}(1.0 + 0.0im), num_steps=500) where {T<:Real}
    function dispersion_eq(k)
        tanh_kH = tanh(k * H)
        lhs = k * tanh_kH * g
        rhs = (-sigma / rho) * k^3 * tanh_kH + omega^2 - 4im * nu * omega * k^2
        return lhs - rhs
    end

    k = k0
    for _ in 1:num_steps
        f = dispersion_eq(k)
        dk = T(1e-8)
        f_plus = dispersion_eq(k + dk)
        f_minus = dispersion_eq(k - dk)
        df = (f_plus - f_minus) / (2 * dk)
        k = k - f / df
    end
    return k
end

end # module
