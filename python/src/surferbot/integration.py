import jax.numpy as jnp

def integration_weights(a, b, N):
    """
    Given an interval [a, b] and N equispaced points,
    return the nodes and quadrature weights such that the quadrature
    is exact for all polynomials of degree <= 3.
    
    Parameters:
      a, b   : float
               Endpoints of the integration interval.
      N      : int
               Number of nodes (N >= 4 for nondegenerate solution).
               
    Returns:
      x      : jnp.ndarray of shape (N,)
               The equispaced nodes.
      w      : jnp.ndarray of shape (N,)
               The quadrature weights satisfying
               sum_{i=0}^{N-1} w[i] * x[i]^k = ∫_a^b x^k dx   for k=0,1,2,3.
    """
    # Generate N equispaced nodes over [a, b].
    x = jnp.linspace(a, b, N)
    
    # Build the Vandermonde-like matrix for monomials 0,1,2,3.
    exponents = jnp.arange(4)
    # A is a 4 x N matrix whose (k,i) entry is x[i]^k.
    A = jnp.vstack([x**k for k in exponents])
    
    # Compute the exact moments:
    # ∫_a^b x^k dx = (b^(k+1)-a^(k+1))/(k+1)  for k = 0, 1, 2, 3.
    moments = jnp.array([(b**(k+1) - a**(k+1))/(k+1) for k in exponents])
    
    # To obtain one set of weights that exactly integrates degree<=3 polynomials,
    # we compute the minimum-norm solution of A @ w = moments.
    # This is given by w = pinv(A) @ moments.
    w = jnp.linalg.pinv(A) @ moments
    return x, w
