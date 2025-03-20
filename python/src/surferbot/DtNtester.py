import jax.numpy as jnp
import scipy.integrate as spi

# Define a harmonic function
def phi(x, y):
    return jnp.log(jnp.sqrt((x)**2+(y-.1)**2))

# Compute vertical derivative (partial derivative w.r.t y) at (x0,0)
x0 = 10.0  # Test point
eps = 0.1
numerical_phi_z = (phi(x0, eps) - phi(x0, -eps))/(2*eps)

# Define the integral function (principal value integral)
def integrand(x):
    return (phi(x0, 0.0) - phi(x, 0.0)) / (x - x0) ** 2

# Integrating near the origin
Dx = eps/2
f_evals = jnp.array([phi(x, 0) for x in [x0-2*Dx, x0-Dx, x0, x0+Dx, x0+2*Dx]])
int_near_x0 = jnp.dot(f_evals, jnp.array([-1.0, -32.0, 66.0, -32.0, -1.0]))/(18*Dx)

# Use SciPy's to integrate. 
LIMIT = 1000
integral_value = spi.quad(integrand, -jnp.inf, x0-eps, limit=LIMIT)[0] + spi.quad(integrand, x0+eps, jnp.inf, limit=LIMIT)[0] + int_near_x0
integral_uncer = spi.quad(integrand, -jnp.inf, x0-eps, limit=LIMIT)[1] + spi.quad(integrand, x0+eps, jnp.inf, limit=LIMIT)[1] 
approx_phi_z = integral_value / jnp.pi

def DtN_generator(Delta_x = 1/jnp.float32(100), N = None):
    '''
    This script will generate the matrix so that Aphi is an approximation of dphi/dz
    '''
    N = int(1/Delta_x) if N is None else N

    # Create the main diagonal with 66's
    DtN1 = jnp.diag(jnp.full(N, 66))
    
    # Fill the first sub- and super-diagonals with -32's
    if N > 1:
        DtN1 += jnp.diag(jnp.full(N-1, -32), k=1)
        DtN1 += jnp.diag(jnp.full(N-1, -32), k=-1)
        
    # Fill the second sub- and super-diagonals with -1's
    if N > 2:
        DtN1 += jnp.diag(jnp.full(N-2, -1), k=2)
        DtN1 += jnp.diag(jnp.full(N-2, -1), k=-2)
        
    DtN1 = DtN1 / 18.0 # This is the integral around the origin
    DtN2 = jnp.diag(jnp.full(N, 1.0)) # First integral away of the origin. 
    
    # Now second integral away from the origin
    coefficients = [0 for _ in range(N+1)]
    coef = lambda n, d: -jnp.float32(n)/(n+d) + (2*n - d)/2 * jnp.log((n+1)/(n-1)) - 1
    for jj in range(1, int(N/2)):
        n = 2 * jj + 1
        coefficients[n-1] += coef(n, -1.0)
        coefficients[n+1] += coef(n, +1.0)
        coefficients[n]   += -2*coef(n, 0.0)

    coefficients = jnp.array(coefficients)  
    i = jnp.arange(N)
    j = jnp.arange(N)
    I, J = jnp.meshgrid(i, j, indexing='ij')
    diff = J - I
    # Use jnp.where to apply the function elementwise
    DtN3 = jnp.where(diff >= 0, 
                  jnp.take(coefficients, diff),  
                  -jnp.take(coefficients, -diff)) 
    
    
    # Apply the function f(i, j) to all pairs (i, j)
    DtN = DtN1 + DtN2 + DtN3  # Broadcasting will handle the rest

    return DtN/(jnp.pi * Delta_x), DtN1/(jnp.pi * Delta_x), DtN2/(jnp.pi * Delta_x), DtN3/(jnp.pi * Delta_x)

A, B, C, D = DtN_generator(N=501, Delta_x=1000)

# Compare results
print(f"Numerical phi_z: {numerical_phi_z}")
print(f"Integral near x0: {int_near_x0} ")
print(f"Integral approximation: {approx_phi_z} +- {integral_uncer/jnp.pi}")
