#import numpy as jnp
from jax import numpy as jnp
from findiff import Diff, coefficients

x = jnp.linspace(0, 10, 100)
dx = x[1] - x[0]
f = jnp.sin(x)
g = jnp.cos(x)

d_dx = Diff(0, dx)

coefficients(deriv=1, acc=2)
d2_dx2 = Diff(0, float(dx), acc=4) ** 2
result = d2_dx2(f)

x, y, z = [jnp.linspace(-1, 1, 5)]*3
dx, dy, dz = float(x[1] - x[0]), float(y[1] - y[0]), float(z[1] - z[0])
X, Y= jnp.meshgrid(x, y, indexing='ij')
boundaries = (jnp.abs(X) == 1) | (jnp.abs(Y) == 1)
f = jnp.sin(X) * jnp.cos(Y) # * jnp.sin(Z)

linear_op =  Diff(0, dx)**2 + Diff(1, dy)**2

print(linear_op.matrix((5, 5)))