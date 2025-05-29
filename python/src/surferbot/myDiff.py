from findiff import Diff as _Diff
from findiff import grids
import numpy as np
import numbers
import types
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from surferbot.constants import SPARSE
from surferbot.sparse_utils import _SparseAtProxy
jsparse.BCOO.at = property(lambda self: _SparseAtProxy(self))


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
    elif isinstance(config_or_axis, (np.ndarray, jnp.ndarray, jsparse.BCOO)):
        return grids.NonEquidistantAxis(dim, coords=config_or_axis, periodic=periodic)

def _dynamic_s_op_method(S, shape=None):
    target_shape = shape if shape is not None else S.shape
    if isinstance(target_shape, tuple) and len(target_shape) <= 2:
        if SPARSE:
            return jsparse.BCOO.from_scipy_sparse(S.matrix(target_shape)).reshape(*(target_shape + target_shape))
        else:
            return jnp.array(S.matrix(target_shape).toarray().reshape(*(target_shape + target_shape)))
    else:
        return NotImplemented


class Diff(_Diff):
    def __init__(self, axis=0, grid=None, shape= None, periodic=False, acc=_Diff.DEFAULT_ACC):
        grid_axis = make_axis(axis, grid, periodic)
        super().__init__(axis, grid_axis, acc)
        self.shape = shape

    # TODO: Write a test to check if this is AD compliant.
    def op(self, shape=None):
        return _dynamic_s_op_method(self, shape=shape)
        
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
        S = super().__mul__(other)
        if self.shape == other.shape:
            S.shape = self.shape
            S.op = types.MethodType(_dynamic_s_op_method, S)
            return S
        return NotImplemented
    
    def __rmul__(self, other):
        if isinstance(other, (numbers.Number, jnp.ndarray)):
            return self.__mul__(other)
        return super().__rmul__(other)
    