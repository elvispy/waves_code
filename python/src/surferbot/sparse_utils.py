import jax.numpy as jnp
from jax.experimental.sparse import BCOO

def _bcoo_set_slice(a: BCOO, idx, value):
    """Return a copy of `a` with a[idx] = value.
       Supported `idx` components: int, slice(None), Ellipsis."""
    # --- normalise idx to length == ndim ----------------------------------
    if not isinstance(idx, tuple):
        idx = (idx,)
    if Ellipsis in idx:                       # expand ...
        e = idx.index(Ellipsis)
        idx = idx[:e] + (slice(None),) * (a.ndim - len(idx) + 1) + idx[e + 1:]
    idx = idx + (slice(None),) * (a.ndim - len(idx))

    # --- build boolean mask of entries to overwrite -----------------------
    mask = jnp.ones(a.data.shape[0], dtype=bool)
    for axis, s in enumerate(idx):
        if isinstance(s, int):
            mask &= (a.indices[:, axis] == s)
        elif isinstance(s, slice):
            if s != slice(None):
                raise NotImplementedError("Only ':' slices supported.")
        else:
            raise NotImplementedError("Index type not supported.")

    # --- overwrite existing stored entries --------------------------------
    data_new = a.data.at[mask].set(value)

    # Optionally drop explicit zeros to keep storage tight
    if value == 0: #TODO: THis is not jit compliant according to ChatGPT. Revise this.
        keep = data_new != 0
        data_new    = data_new[keep]
        indices_new = a.indices[keep]
    else:
        indices_new = a.indices                   # don’t add new coords

    return BCOO((data_new, indices_new), shape=a.shape)


class _SparseAtProxy:
    def __init__(self, arr):  self.arr = arr
    def __getitem__(self, idx): self.idx = idx; return self
    def set(self, val):         return _bcoo_set_slice(self.arr, self.idx, val)

if __name__ == "__main__":
    
    # attach as a read-only property
    BCOO.at = property(lambda self: _SparseAtProxy(self))


    # Build a sample sparse (3,3) tensor with one non-zero per row
    dense   = jnp.arange(1, 10).reshape(3, 3).astype(float)
    A       = BCOO.fromdense(dense * (dense % 2 == 1))    # keep odds only

    # 1) zero the middle column
    A = A.at[:, 1].set(0.0)

    # 2) set the first row to a constant 5 (only stored entries touched)
    A = A.at[0, :].set(5.0)

    print(A.todense())
    # [[5. 0. 5.]
    #  [0. 0. 0.]
    #  [7. 0. 9.]]
