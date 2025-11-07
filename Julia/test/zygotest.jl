using Zygote
using ChainRules
using LinearAlgebra
using SparseArrays
#-------------------------------------------------------
function test_func(x)

    #
    n = length(x)

    # make matrix
    A_buf = Zygote.Buffer(spzeros(Float64, n, n))
    k_tmp = rand(Float64, n, n)
    for i in 1 : n
        for j in 1 : n
            A_buf[i, j] += k_tmp[i, j] + x[j]^2.0
        end
    end

    # Zygote.Buffer -> SparseMatrixCSC
    A = copy(A_buf)

    # righ hand side
    b = fill(1.0 , n)

    # solve !!!!
    c = A \ b

    # compute evaluate function value
    return norm(c)
end
#-------------------------------------------------------
s = rand(Float64, 30)
@time value = test_func(s)

df_auto(x) = test_func'(x)
@time display(df_auto(s))