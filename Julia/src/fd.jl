module FD

using SparseArrays
using LinearAlgebra
using ForwardDiff

export getNonCompactFDMWeights, getNonCompactFDmatrix, getNonCompactFDmatrix2D

"""
    checkInput(n::Int, stencil::AbstractVector{<:Real})

Validate the stencil for a given derivative order `n`.

# Arguments
- `n`: Order of the derivative.
- `stencil`: Vector of relative grid indices.

# Returns
- The stencil as a vector.
"""
function checkInput(n::Int, stencil::AbstractVector{<:Real})
    stencil_vec = collect(stencil)
    s = length(stencil_vec)
    if s <= n
        throw(ArgumentError("The stencil size should be larger than n."))
    end
    L = minimum(stencil_vec)
    U = maximum(stencil_vec)
    if L > 0 || U < 0 || (U - L) < 1
        throw(ArgumentError("The stencil is invalid."))
    end
    return stencil_vec
end

"""
    getNonCompactFDMWeights(dx::Real, n::Int, stencil::AbstractVector{<:Real})

Compute finite difference weights for the `n`-th derivative on an arbitrary stencil.

# Arguments
- `dx`: Grid spacing.
- `n`: Order of the derivative.
- `stencil`: Vector of relative grid indices.

# Returns
- A tuple `(weights, ooa)` where `weights` is the vector of coefficients and `ooa` is the order of accuracy.
"""
function getNonCompactFDMWeights(dx::Real, n::Int, stencil::AbstractVector{<:Real})
    stencil_vec = checkInput(n, stencil)
    s = length(stencil_vec)
    T = typeof(dx)
    sig = zeros(T, s, s)
    for i in 0:(s - 1)
        for j in 1:s
            sig[i + 1, j] = T(stencil_vec[j]^i / factorial(i))
        end
    end
    rhs = zeros(T, s)
    rhs[n + 1] = one(T)
    
    # Solver that is safe for Duals in the matrix
    a = if T <: ForwardDiff.Dual
        # Separating value and partials for the small sig \ rhs solve
        sig0 = ForwardDiff.value.(sig)
        rhs0 = ForwardDiff.value.(rhs)
        a0 = sig0 \ rhs0
        
        p = ForwardDiff.npartials(sig[1,1])
        a_part = zeros(eltype(a0), s, p)
        for p_idx in 1:p
            sig_p = ForwardDiff.partials.(sig, p_idx)
            rhs_p = ForwardDiff.partials.(rhs, p_idx)
            a_part[:, p_idx] = sig0 \ (rhs_p - sig_p * a0)
        end
        [T(a0[i], ForwardDiff.Partials(Tuple(a_part[i, :]))) for i in 1:s]
    else
        sig \ rhs
    end
    
    w = a' ./ (dx^n)

    ooa = s - n
    extraEq = [T(stencil_vec[j]^s / factorial(s)) for j in 1:s]
    tol = eps(ForwardDiff.value(T(1.0))) * 1e4
    if abs(sum(extraEq .* a)) < tol
        ooa += 1
    end
    return collect(w), ooa
end

"""
    getNonCompactFDmatrix(npx::Int, dx::Real, n::Int, ooa::Int)

Assemble a sparse finite difference matrix for the `n`-th derivative.

# Arguments
- `npx`: Number of grid points.
- `dx`: Grid spacing.
- `n`: Order of the derivative.
- `ooa`: Requested order of accuracy.

# Returns
- A sparse matrix of size `npx` x `npx`.
"""
function getNonCompactFDmatrix(npx::Int, dx::T, n::Int, ooa::Int) where {T<:Real}
    ooa_eff = isodd(ooa) ? ooa + 1 : ooa
    s = ooa_eff + n
    sC = iseven(n) ? s - 1 : s
    if npx < s
        throw(ArgumentError("Not enough grid points for the specified order of accuracy."))
    end

    U = (sC - 1) ÷ 2
    stencilC = collect(-U:U)
    wC, ooaCal = getNonCompactFDMWeights(dx, n, stencilC)
    if ooaCal != ooa_eff
        throw(ArgumentError("Order of accuracy mismatch."))
    end

    D = spzeros(T, npx, npx)
    for (i, offset) in enumerate(stencilC)
        coeff = wC[i]
        if offset >= 0
            for row in 1:(npx - offset)
                D[row, row + offset] = coeff
            end
        else
            for row in (1 - offset):npx
                D[row, row + offset] = coeff
            end
        end
    end

    stencil0 = collect(0:(s - 1))
    mid = (s - 1) / 2
    for ind in stencil0
        if ind == mid
            continue
        end
        shifted = stencil0 .- ind
        w, ooaCal = getNonCompactFDMWeights(dx, n, shifted)
        if ooaCal != ooa_eff
            throw(ArgumentError("Order of accuracy mismatch."))
        end

        if ind < mid
            row = ind + 1
            D[row, 1:s] = sparse(reshape(w, 1, s))
        elseif ind > mid
            row = npx - (s - ind - 1)
            D[row, (end - s + 1):end] = sparse(reshape(w, 1, s))
        end
    end

    return D
end

"""
    getNonCompactFDmatrix2D(npx::Int, npy::Int, dx::Real, dy::Real, n::Int, ooa::Int)

Assemble 2D sparse finite difference matrices for the `n`-th partial derivatives.

# Arguments
- `npx`, `npy`: Grid dimensions.
- `dx`, `dy`: Grid spacings.
- `n`: Order of the derivative.
- `ooa`: Requested order of accuracy.

# Returns
- A tuple `(Dx2D, Dy2D)` of sparse matrices.
"""
function getNonCompactFDmatrix2D(npx::Int, npy::Int, dx::T, dy::T, n::Int, ooa::Int) where {T<:Real}
    Dx1D = getNonCompactFDmatrix(npx, dx, n, ooa)
    Dy1D = getNonCompactFDmatrix(npy, dy, n, ooa)
    Inpy = sparse(one(T) * I, npy, npy)
    Inpx = sparse(one(T) * I, npx, npx)
    Dx2D = kron(Dx1D, Inpy)
    Dy2D = kron(Inpx, Dy1D)
    return Dx2D, Dy2D
end

end
