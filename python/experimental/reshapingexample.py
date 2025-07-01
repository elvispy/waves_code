import jax.numpy as jnp

# Dimensions
N, M = 4, 3

# 1) Build the 6th-order tensor A6 and the “true” x array
A6 = jnp.arange(5 * N * M * N * M * 5).reshape((5, N, M, N, M, 5))
x_true = jnp.arange(N * M * 5).reshape((N, M, 5))

# 2) Flatten into a 2D matrix and a vector
rows = cols = 5 * N * M
A2 = A6.reshape((rows, cols))
x_vec = x_true.ravel()

# 3) Compute b in two ways
b_from_A2        = (A2 @ x_vec).reshape((5, N, M))
b_from_tensordot = jnp.tensordot(A6, x_true, axes=([3,4,5], [0,1,2]))

# 4) Check they match
print("b_from_A2:\n", b_from_A2)
print("\nb_from_tensordot:\n", b_from_tensordot)
print("\nAll close?", jnp.allclose(b_from_A2, b_from_tensordot))
