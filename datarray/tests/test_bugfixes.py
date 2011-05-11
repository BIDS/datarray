import numpy as np

from datarray.datarray import Axis, DataArray, NamedAxisError, \
    _pull_axis, _reordered_axes

from datarray.testing.utils import assert_datarray_equal

import nose.tools as nt
import numpy.testing as npt

def test_bug3():
    "Bug 3"
    x = np.array([1,2,3])
    y = DataArray(x, 'x')
    nt.assert_equal( x.sum(), y.sum() )
    nt.assert_equal( x.max(), y.max() )

def test_bug44():
    "Bug 44"
    # In instances where axis=None, the operation runs
    # on the flattened array. Here it makes sense to return
    # the op on the underlying np.ndarray.
    A = [[1,2,3],[4,5,6]]
    x = DataArray(A, 'xy').std()
    y = np.std(A)
    nt.assert_equal( x.sum(), y.sum() )

def test_bug45():
    "Bug 45"
    A = DataArray([1,2,3], 'a'); B = DataArray([2,3,4], 'b'); C = np.outer(A,B)
    assert_datarray_equal(C,DataArray(C, 'ab'))

