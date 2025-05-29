import jax
import jax.numpy as jnp
import math
import jax.numpy as jnp
from findiff import Diff as _Diff
from findiff import grids
import numpy as np
import numbers
from integration import simpson_weights
 
def make_axis(dim, config_or_axis, periodic=False):
    """
    A reimplementation of the make_axis function to create a grid axis.
    Supports Jax arrays
    """
    if isinstance(config_or_axis, grids.GridAxis):
        return config_or_axis
    if isinstance(config_or_axis, numbers.Number):
        return grids.EquidistantAxis(dim, spacing=config_or_axis, periodic=periodic)
    elif isinstance(config_or_axis, jnp.ndarray) and len(config_or_axis) == 1:
        return grids.EquidistantAxis(dim, spacing=config_or_axis.item(0), periodic=periodic)
    elif isinstance(config_or_axis, (np.ndarray, jnp.ndarray)):
        return grids.NonEquidistantAxis(dim, coords=config_or_axis, periodic=periodic)


class Diff(_Diff):

    def __init__(self, axis=0, grid=None, shape= None, periodic=False, acc=_Diff.DEFAULT_ACC):
        grid_axis = make_axis(axis, grid, periodic)
        super().__init__(axis, grid_axis, acc)
        self.shape = shape

    # TODO: Implement an instatiatior to make AD compliant. And check this implementation
    def op(self, shape=None):
        if shape is None:
            if self.shape is None:
                raise ValueError("Shape must be provided if not set in the constructor.")
            shape = self.shape
        if isinstance(shape, tuple) and len(shape) <= 2:
            return jnp.array(self.matrix(shape).toarray().reshape(*(shape + shape)))
        else:
            return NotImplemented
        
    def __pow__(self, power):
        """Returns a Diff instance for a higher order derivative."""
        new_diff = Diff(self.dim, self.axis, shape=self.shape, acc=self.acc)
        new_diff._order *= power
        return new_diff

    def __mul__(self, other):
        if isinstance(other, Diff) and self.dim == other.dim:
            new_diff = Diff(self.dim, self.axis, shape=self.shape, acc=self.acc)
            new_diff._order += other.order
            return new_diff
        elif isinstance(other, (numbers.Number, jnp.ndarray)):
            return other * self.op()
        return super().__mul__(other)
    
    def __rmul__(self, other):
        if isinstance(other, (numbers.Number, jnp.ndarray)):
            return self.__mul__(other)
        return super().__rmul__(other)


def build_matrix(params):
    a, b, c = params
    return jnp.array([
        [a,   1.0 + 0j, 2.0 + 0j],
        [0.0 + 0j, b,   1.0 + 0j],
        [1.0 + 0j, 1.0 + 0j, c],
        [2.0 + 0j, 0.0 + 0j, 1.0 + 0j],
    ], dtype=jnp.complex64)

def build_vector(params):
    a, b, c = params
    return jnp.array([1.0 + a, 2.0 + b, 3.0 + c, 4.0], dtype=jnp.complex64)

def objective(params):
    d_dx = Diff(axis=0, grid=1.0, shape=(4,))
    A = params[0] * d_dx + params[1] * (d_dx ** 2) - params[3]
    b = build_vector(params)
    x = jnp.linalg.lstsq(A, b, rcond=None)[0]
    return jnp.real(jnp.sum(x))  # real scalar output for grad

# Must split real and imaginary parts if you want complex params
# Here we treat real-valued parameters that produce complex A and b
grad_fn = jax.grad(objective)

params = jnp.array([10.0, 20.0, 30.0])
print("Loss (sum of x):", objective(params))
print("Gradient:", grad_fn(params))

