import numpy as np

from datarray.datarray import Axis, DataArray, NamedAxisError, \
    _pull_axis, _reordered_axes

from datarray.testing.utils import assert_datarray_equal
import datarray.print_grid as print_grid

import nose.tools as nt
import numpy.testing as npt

def test_full_reduction():
    # issue #2
    nt.assert_equal(DataArray([1, 2, 3]).sum(axis=0),6)

def test_bug3():
    "Bug 3"
    x = np.array([1,2,3])
    y = DataArray(x, 'x')
    nt.assert_equal( x.sum(), y.sum() )
    nt.assert_equal( x.max(), y.max() )

def test_1d_label_indexing():
    # issue #18
    cap_ax_spec = 'capitals', ['washington', 'london', 'berlin', 'paris', 'moscow']
    caps = DataArray(np.arange(5),[cap_ax_spec])
    caps.axes.capitals["washington"]

def test_bug26():
    "Bug 26: check that axes names are computed on demand."
    a = DataArray([1,2,3])
    nt.assert_true(a.axes[0].name is None)
    a.axes[0].name = "a"
    nt.assert_equal(a.axes[0].name, "a")

def test_bug35():
    "Bug 35"
    txt_array = DataArray(['a','b'], axes=['dummy'])
    #calling datarray_to_string on string arrays used to fail
    print_grid.datarray_to_string(txt_array)
    #because get_formatter returned the class not an instance
    assert isinstance(print_grid.get_formatter(txt_array),
                      print_grid.StrFormatter)

def test_bug38():
    "Bug 38: DataArray.__repr__ should parse as a single entity"
    # Calling repr() on an ndarray prepends array (instead of np.array)
    array = np.array
    arys = (
        DataArray(np.random.randint(0, 10000, size=(1,2,3,4,5)), 'abcde'),
        DataArray(np.random.randint(0, 10000, size=(3,3,3))), # Try with missing axes
        DataArray(np.random.randint(0, 10000, (2,4,5,6)), # Try with ticks
            ('a', ('b', ('b1','b2','b3','b4')), 'c', 'd')),
        )
    for A in arys:
        print A
        assert_datarray_equal(A, eval(repr(A)))

def test_bug44():
    "Bug 44"
    # In instances where axis=None, the operation runs
    # on the flattened array. Here it makes sense to return
    # the op on the underlying np.ndarray.
    A = [[1,2,3],[4,5,6]]
    x = DataArray(A, 'xy').std()
    y = np.std(A)
    nt.assert_equal( x.sum(), y.sum() )

