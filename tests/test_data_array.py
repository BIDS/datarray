''' Tests for DataArray and friends '''

import numpy as np

from datarray.datarray import Axis, DataArray, NamedAxisError, \
    _pull_axis, _reordered_axes

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
    b0 = b.axis.x[0]
    yield npt.assert_equal, b0, [1,2]
    # slice slicing
    b1 = b.axis.x[1:]
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
    yield (nt.assert_true,isinstance(a.axis.x[0],int))
    # Verify indexing of axis
    yield (nt.assert_equals, a.axis.x.index, 0)
    # Iteration checks
    for i,val in enumerate(a.axis.x):
        yield (nt.assert_equals,val,adata[i])
        yield (nt.assert_true,isinstance(val,int))


def test_2d():
    b = DataArray([[1,2],[3,4],[5,6]], 'xy')
    yield (nt.assert_equals, b.names, ['x','y'])
    # Check row named slicing
    rs = b.axis.x[0]
    yield (npt.assert_equal, rs, [1,2])
    yield nt.assert_equal, rs.names, ['y']
    yield nt.assert_equal, rs.axes, [Axis('y', 0, rs)]
    # Now, check that when slicing a row, we get the right names in the output
    yield (nt.assert_equal, b.axis.x[1:].names, ['x','y'])
    # Check column named slicing
    cs = b.axis.y[1]
    yield (npt.assert_equal, cs, [2,4,6])
    yield nt.assert_equal, cs.names, ['x']
    yield nt.assert_equal, cs.axes, [Axis('x', 0, cs)]
    # What happens if we do normal slicing?
    rs = b[0]
    yield (npt.assert_equal, rs, [1,2])
    yield nt.assert_equal, rs.names, ['y']
    yield nt.assert_equal, rs.axes, [Axis('y', 0, rs)]
    

def test__pull_axis():
    a = Axis('x', 0, None)
    b = Axis('y', 1, None)
    c = Axis('z', 2, None)
    t_pos = Axis('y', 1, None)
    t_neg = Axis('x', 5, None)
    axes = [a, b, c]
    yield nt.assert_true, t_pos in axes
    yield nt.assert_false, t_neg in axes
    yield nt.assert_equal, axes, _pull_axis(axes, t_neg)
    yield nt.assert_equal, axes[:-1], _pull_axis(axes, c)
    new_axes = [a, Axis('z', 1, None)]
    yield nt.assert_equal, new_axes, _pull_axis(axes, t_pos)
    

def test__reordered_axes():
    a = Axis('x', 0, None)
    b = Axis('y', 1, None)
    c = Axis('z', 2, None)
    res = _reordered_axes([a,b,c], (1,2,0))
    names_inds = [(ax.name, ax.index) for ax in res]
    yield nt.assert_equal, set(names_inds), set([('y',0),('z',1),('x',2)])
    
