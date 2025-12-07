using FiniteDifferences

println("Forward 2, 1: ", forward_fdm(2, 1).coefs)
println("Forward 3, 1: ", forward_fdm(3, 1).coefs)
println("Backward 2, 1: ", backward_fdm(2, 1).coefs)
