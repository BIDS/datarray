''' Tests for dataarray '''

import numpy as np

from datarray import Axis, DataArray, NamedAxisError

import nose.tools as nt
import numpy.testing as npt


def test_axis_equal():
    ax1 = Axis('aname', 0, None)
    ax2 = Axis('aname', 0, None)
    yield nt.assert_equal, ax1, ax2
    # The array to which the axis points does not matter in comparison
    ax3 = Axis('aname', 0, np.arange(10))
    yield nt.assert_equal, ax1, ax3
    # but the index does
    ax4 = Axis('aname', 1, None)
    yield nt.assert_not_equal, ax1, ax4
    # so does the name
    ax5 = Axis('anothername', 0, None)
    yield nt.assert_not_equal, ax1, ax5
    # and obviously both
    yield nt.assert_not_equal, ax4, ax5
    

def test_basic():
    adata = [2,3]
    a = DataArray(adata, 'x', float)
    yield nt.assert_equal, a.names, ['x']
    yield nt.assert_equal, a.dtype, np.dtype(float)
    b = DataArray([[1,2],[3,4],[5,6]], 'xy')
    yield nt.assert_equal, b.names, ['x','y']
    # integer slicing
    b0 = b.ax_x[0]
    yield npt.assert_equal, b0, [1,2]
    # slice slicing
    b1 = b.ax_x[1:]
    yield npt.assert_equal, b1, [[3,4], [5,6]]


def test_combination():
    narr = DataArray(np.zeros((1,2,3)), names=('a','b','c'))
    n3 = DataArray(np.ones((1,2,3)), names=('x','b','c'))
    yield nt.assert_raises, NamedAxisError, np.add, narr, n3
    # addition of scalar
    res = narr + 2
    yield nt.assert_true, isinstance(res, DataArray)
    yield nt.assert_equal, res.axes, narr.axes
    # addition of matching size array, with matching names
    res = narr + narr
    yield nt.assert_equal, res.axes, narr.axes


def test_1d():
    adata = [2,3]
    a = DataArray(adata, 'x', int)
    # Verify scalar extraction
    yield (nt.assert_true,isinstance(a.ax_x[0],int))
    # Verify indexing of axis
    yield (nt.assert_equals, a.ax_x.index, 0)
    # Iteration checks
    for i,val in enumerate(a.ax_x):
        yield (nt.assert_equals,val,adata[i])
        yield (nt.assert_true,isinstance(val,int))


def test_2d():
    b = DataArray([[1,2],[3,4],[5,6]], 'xy')
    yield (nt.assert_equals, b.names, ['x','y'])
    # Check row slicing
    yield (npt.assert_equal, b.ax_x[0], [1,2])
    # Check column slicing
    yield (npt.assert_equal, b.ax_y[1], [2,4,6])
    # Now, check that when slicing a row, we get the right names in the output
    yield (nt.assert_equal, b.ax_x[1:].names, ['x','y'])
    yield (nt.assert_equal, b.ax_x[0].names, ['y'])
