using SparseArrays, LinearAlgebra
using LinearSolve              # Krylov methods
using ForwardDiff              # for ∂b/∂x and (∂A/∂x)·y via AD

# --- Toy sparse pattern (scale n up as you like)
function laplace1d(n)
    I = Int[]
    J = Int[]
    V = Float64[]
    for i in 1:n
        push!(I, i); push!(J, i); push!(V, 2.0)
        if i > 1
            push!(I, i); push!(J, i-1); push!(V, -1.0)
        end
        if i < n
            push!(I, i); push!(J, i+1); push!(V, -1.0)
        end
    end
    sparse(I, J, V, n, n)
end

# Basis blocks S_i (here three localized patterns; in practice you have many)
function basis_blocks(n)
    S1 = laplace1d(n)                           # global-like
    S2 = spdiagm(0 => [sin(0.01*i) for i=1:n])  # diagonal pattern
    S3 = spdiagm(0 => [i%7==0 ? 1.0 : 0.0 for i=1:n])  # sparse diagonal
    (S1, S2, S3)
end

# Nonlinear A(x): A = L + x1*S1 + exp(x2)*S2 + sin(x3)*S3
function A_of(x, S, L)
    S1,S2,S3 = S
    L + x[1]*S1 + exp(x[2])*S2 + sin(x[3])*S3
end

# Nonlinear b(x)
b_of(x) = @views [x[1]^2; x[1]*x[2]; exp(x[3]); zeros(length(x)+97)]  # pad to length n as needed

# Helper to size b to n
function b_n_of(x, n)
    b = b_of(x)
    if length(b) < n
        return vcat(b, zeros(n - length(b)))
    else
        return b[1:n]
    end
end

# Solve y(x) = A(x)\b(x) using iterative Krylov + simple Jacobi preconditioner
function solve_y(x, S, L; tol=1e-8, maxiters=10_000)
    A = A_of(x, S, L)
    b = b_n_of(x, size(A,1))
    # Jacobi preconditioner (placeholder; replace with AMG/ILU in production)
    D = diag(A)
    M = Diagonal(1.0 ./ max.(abs.(D), 1e-12))

    prob = LinearProblem(A, b)
    sol = solve(prob, KrylovJL_GMRES(); Pl=M, reltol=tol, maxiters=maxiters, verbose=false)
    return sol.u, A, M
end

# ---------- Option A: columns J_i (few parameters p) ----------
# dy/dx_i via implicit rule with iterative solves
function jacobian_columns(x, S, L)
    y, A, M = solve_y(x, S, L)
    n = length(y); p = length(x)
    J = zeros(n, p)

    # AD for Jb = ∂b/∂x, and JAy = (∂A/∂x)·y via AD on g(x)=A(x)*y
    Jb  = ForwardDiff.jacobian(x -> b_n_of(x, n), x)
    g   = x_ -> A_of(x_, S, L) * y
    JAy = ForwardDiff.jacobian(g, x)

    # For each i: solve A * J_i = Jb[:,i] - JAy[:,i]
    for i in 1:p
        rhs = @views Jb[:,i] .- JAy[:,i]
        prob = LinearProblem(A, rhs)
        sol  = solve(prob, KrylovJL_GMRES(); Pl=M, reltol=1e-8, maxiters=10_000, verbose=false)
        @views J[:,i] .= sol.u
    end
    return y, J
end

# ---------- Option B: Jᵀ·v via one adjoint solve (preferred when p is large) ----------
# For φ(x)=vᵀ y(x), ∇φ = (∂b/∂x)ᵀ w - (∂A/∂x : w yᵀ) where Aᵀ w = v
function JT_v(x, v, S, L)
    y, A, M = solve_y(x, S, L)

    # Adjoint solve: Aᵀ w = v
    probT = LinearProblem(A', v)       # same preconditioner structure; tweak if nonsymmetric
    solT  = solve(probT, KrylovJL_GMRES(); Pl=M, reltol=1e-8, maxiters=10_000, verbose=false)
    w = solT.u

    # Compute (∂b/∂x)ᵀ w via AD without forming full Jacobian:
    # Use ForwardDiff to get Jb, then Jb' * w (OK for small p).
    n = length(y)
    Jb = ForwardDiff.jacobian(x -> b_n_of(x, n), x)
    term_b = Jb' * w

    # Compute (∂A/∂x : w yᵀ) columns via AD on h_i(x) = A(x) y, then dot with w:
    g = x_ -> A_of(x_, S, L) * y
    JAy = ForwardDiff.jacobian(g, x)   # columns (∂A/∂x_i) y
    term_A = map(i -> dot(w, @view JAy[:,i]), 1:length(x))  # wᵀ (∂A/∂x_i) y

    # Gradient of φ = Jᵀ v
    return term_b .- term_A, y, w
end

# ----------------- Run tiny PoC -----------------
n = 20              # toy size; structure scales to large n
L = laplace1d(n)
S = basis_blocks(n)
x = [0.4, -0.3, 0.2]
v = randn(n)         # seed for Jᵀ v test

# Columns (for small p)
y, J = jacobian_columns(x, S, L)
println("‖J_columns‖ = ", opnorm(J))

# Adjoint Jᵀ v (one extra solve)
gJT, y2, w = JT_v(x, v, S, L)
println("length(Jᵀ v) = ", length(gJT))
