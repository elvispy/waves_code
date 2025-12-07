using FiniteDifferences

m = central_fdm(2, 1)
println("Type: ", typeof(m))
println("Field names: ", fieldnames(typeof(m)))
if hasfield(typeof(m), :grid)
    println("Grid: ", m.grid)
end
if hasfield(typeof(m), :coefs)
    println("Coefs: ", m.coefs)
end

m_forward = forward_fdm(2, 1)
println("Forward Grid: ", m_forward.grid)
println("Forward Coefs: ", m_forward.coefs)
