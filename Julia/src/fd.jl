module FD

using SparseArrays
using LinearAlgebra

export getNonCompactFDMWeights, getNonCompactFDmatrix, getNonCompactFDmatrix2D

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

function getNonCompactFDMWeights(dx::Real, n::Int, stencil::AbstractVector{<:Real})
    stencil_vec = checkInput(n, stencil)
    s = length(stencil_vec)
    sig = zeros(Float64, s, s)
    for i in 0:(s - 1)
        for j in 1:s
            sig[i + 1, j] = stencil_vec[j]^i / factorial(i)
        end
    end
    rhs = zeros(Float64, s)
    rhs[n + 1] = 1.0
    a = sig \ rhs
    w = a' ./ (float(dx)^n)

    ooa = s - n
    extraEq = [stencil_vec[j]^s / factorial(s) for j in 1:s]
    tol = eps() * 1e4
    if abs(sum(extraEq .* a)) < tol
        ooa += 1
    end
    return collect(w), ooa
end

function getNonCompactFDmatrix(npx::Int, dx::Real, n::Int, ooa::Int)
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

    D = spzeros(Float64, npx, npx)
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

    return sparse(D)
end

function getNonCompactFDmatrix2D(npx::Int, npy::Int, dx::Real, dy::Real, n::Int, ooa::Int)
    Dx1D = getNonCompactFDmatrix(npx, dx, n, ooa)
    Dy1D = getNonCompactFDmatrix(npy, dy, n, ooa)
    Dx2D = kron(Dx1D, sparse(I, npy, npy))
    Dy2D = kron(sparse(I, npx, npx), Dy1D)
    return sparse(Dx2D), sparse(Dy2D)
end

end
