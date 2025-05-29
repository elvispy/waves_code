import jax
import jax.numpy as jnp
from jax import grad
from jax.experimental.sparse import BCOO
from jax.scipy.sparse.linalg import gmres  # Conjugate gradient solver

# Create a sparse symmetric positive definite matrix in BCOO format
def create_sparse_matrix(n):
    """Creates a simple SPD sparse matrix A (diagonal + off-diagonal structure)."""
    row = jnp.arange(n)
    col = jnp.arange(n)
    data = 2.0 * jnp.ones(n)  # Diagonal elements

    # Add -1 to off-diagonal positions
    if n > 1:
        row = jnp.concatenate([row, jnp.arange(n - 1), jnp.arange(1, n)])
        col = jnp.concatenate([col, jnp.arange(1, n), jnp.arange(n - 1)])
        data = jnp.concatenate([data, -1.0 * jnp.ones(2 * (n - 1))])

    indices = jnp.stack([row, col], axis=1)  # Shape [nnz, 2]
    mat = BCOO((data, indices), shape=(n, n))

    return mat

# Define a function that solves A x = b
def solve_system(b):
    A = create_sparse_matrix(len(b))
    x, _ = gmres(A, b, maxiter=100)
    return x

# A differentiable loss function based on the solution
def loss_fn(b):
    x = solve_system(b)
    return jnp.sum(x**2)

# Evaluate the gradient of the loss w.r.t. b
n = 10
b = jnp.ones(n)
loss_grad = grad(loss_fn)(b)

print("Gradient of the loss w.r.t. b:", loss_grad)
