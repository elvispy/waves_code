import jax.numpy as jnp

def DtN_generator(N=100):
    '''
    This script will generate the matrix so that Aphi is an approximation of dphi/dz
    '''
    Delta_x = 1/jnp.float32(N)

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
        
    DtN = DtN / (18*jnp.pi * Delta_x) # This is the integral around the origin
    DtN = DtN + jnp.diag(jnp.full(N, 1/(Delta_x*jnp.pi))) # First integral away of the origin. 
    
    # Now second integral away from the origin
    coefficients = [0 for _ in range(N+1)]
    coef = lambda n, d: -jnp.float32(n)/(n+d) + (2*n -d)/2 * jnp.log((n+1)/(n-1)) - 1
    for jj in range(1, int(N/2)):
        n = 2 * jj + 1
        coefficients[n-1] += coef(n, -1)
        coefficients[n+1] += coef(n, +1)
        coefficients[n]   += 2 - 2 * n * jnp.log((n+1)/(n-1))

    coefficients = jnp.array(coefficients)  
    i = jnp.arange(N)
    j = jnp.arange(N)
    I, J = jnp.meshgrid(i, j, indexing='ij')
    diff = J - I
    # Use jnp.where to apply the function elementwise
    matrix = jnp.where(diff >= 0, 
                  jnp.take(coefficients, diff),  # coefficients[J - I] safely
                  -jnp.take(coefficients, -diff))  # -coefficients[I - J] safely
    
    
    # Apply the function f(i, j) to all pairs (i, j)
    DtN = DtN + matrix/(jnp.pi * Delta_x)  # Broadcasting will handle the rest

    return DtN

# Example usage:
if __name__ == "__main__":
    N = 6  # Example size
    matrix = DtN_generator(N)
    print(matrix)

