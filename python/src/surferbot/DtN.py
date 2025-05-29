import jax.numpy as jnp

def DtN_generator(N: int, h: float = 1.0):
    '''
    This script will generate the matrix M so that M @ phi is an approximation of
    \frac{1}{\pi} \left(\lim_{\epsilon\to 0} \int_{|x-x_0| > \epsilon} 
    \frac{\phi(x_0, 0) - \phi(x, 0)}{(x-x_0)^2} dx\right)

    For harmonic functions in the plane with decaying behaviour, this is exactly 
    d/dz phi(x, 0)|_{x = x0}
    '''
    #N = int(1/Delta_x) if N is None else N

    # Create the main diagonal with 66's
    DtN = jnp.diag(jnp.full(N, 66))
    
    # Fill the first sub- and super-diagonals with -32's
    if N > 1:
        DtN += jnp.diag(jnp.full(N-1, -32), k=1)
        DtN += jnp.diag(jnp.full(N-1, -32), k=-1)
        
    # Fill the second sub- and super-diagonals with -1's
    if N > 2:
        DtN += jnp.diag(jnp.full(N-2, -1), k=2)
        DtN += jnp.diag(jnp.full(N-2, -1), k=-2)
        
    DtN = DtN / 18.0 # This is the integral around the origin
    DtN = DtN + jnp.diag(jnp.full(N, 1.0)) # First integral away of the origin. 
    
    # Now second integral away from the origin
    coefficients = [0 for _ in range(N+1)]
    coef = lambda n, d: -jnp.float32(n)/(n+d) + (2*n - d)/2 * jnp.log((n+1)/(n-1)) - 1
    for jj in range(1, int(N/2)):
        n = 2 * jj + 1
        coefficients[n-1] += coef(n, -1.0)
        coefficients[n+1] += coef(n, +1.0)
        coefficients[n]   += -2*coef(n, 0.0)

    coefficients = jnp.array(coefficients)  
    i, j = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
    DtN3 = coefficients[jnp.abs(i - j)]
    
    DtN = DtN + DtN3  # Broadcasting will handle the rest

    return h * DtN/(jnp.pi)

# Example usage:
if __name__ == "__main__":
    N = 6  # Example size
    matrix = DtN_generator(N = N)
    print(matrix)

