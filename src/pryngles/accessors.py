import numpy as np
import pandas as pd

from pryngles.consts import SPANGLER_VEC_GROUPS


# Custom DataFrame accessor providing ergonomic shorthand for vector groups.
# Usage:  df.vectors.center_ecl  ->  (N,3) sub-DataFrame
#        df.vectors.center_ecl.to_numpy()  ->  (N,3) ndarray
@pd.api.extensions.register_dataframe_accessor("vectors")
class SpanglerVectorAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __getattr__(self, name):
        cols = SPANGLER_VEC_GROUPS.get(name)
        if cols is None:
            raise AttributeError(
                f"'{name}' is not a known spangler vector group. Available groups: {sorted(SPANGLER_VEC_GROUPS)}"
            )
        return self._obj[cols]


# ``df.masked.get(cols, mask)`` / ``df.masked.put(cols, values, mask)``: masked numpy reads and writes.
#
# Import this module once (e.g. in your package's __init__) and every DataFrame gets the accessor.
#
# cols:  a column name (-> 1D array), a list of names (-> 2D array), or None for ALL columns, in frame order.
# mask:  None (all rows) or a boolean array / boolean Series of length len(df).
#        An all-True mask is detected automatically and takes the fast path (no masking, often a view).
# Values travel as numpy arrays and are written by POSITION, never aligned on the index.
# This is access only: it never creates columns or changes dtypes (do that yourself, once, up front).
@pd.api.extensions.register_dataframe_accessor("masked")
class MaskedAccessor:
    def __init__(self, df):
        self._df = df

    @staticmethod
    def norm_mask(mask, n):
        """None for 'all rows', else a validated boolean ndarray. Call once and reuse in hot loops."""
        if mask is None:
            return None
        m = mask.to_numpy() if isinstance(mask, pd.Series) else np.asarray(mask)
        if m.dtype != bool or m.shape != (n,):
            raise ValueError(f"mask must be a boolean array of shape ({n},), got {m.dtype} {m.shape}")
        return None if m.all() else m

    def get(self, mask=None, cols=None, dtype=None):
        """Read column(s) as numpy. str -> (k,), list or None -> (k, ncols).

        With cols=None on a frame of mixed dtypes the result is an object array; pass dtype=float
        (or select numeric columns) if you want a float array.
        """
        df = self._df
        m = self.norm_mask(mask, len(df))
        if cols is None:
            src = df if m is None else df.loc[m]
        else:
            src = df[cols] if m is None else df.loc[m, cols]
        return src.to_numpy(dtype=dtype)

    def put(self, mask=None, cols=None, values=None):
        """Write values into column(s) (cols=None -> all columns), by position.

        Pure access: no column creation and no dtype changes. Columns must already exist and have a
        dtype that can hold `values`; otherwise pandas raises (e.g. floats into an int column).
        """
        df = self._df
        m = self.norm_mask(mask, len(df))
        if isinstance(values, (pd.Series, pd.DataFrame)):
            values = values.to_numpy()
        key = list(df.columns) if cols is None else cols
        if np.ndim(values) == 2:
            ncols = 1 if isinstance(key, str) else len(key)
            if np.shape(values)[1] != ncols:
                raise ValueError(f"values has {np.shape(values)[1]} columns but {ncols} column(s) were selected")
        if m is None:
            df[key] = values
        else:
            df.loc[m, key] = values
