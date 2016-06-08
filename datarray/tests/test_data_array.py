'''Tests for DataArray and friend'''

import sys
PY3 = sys.version_info[0] >= 3

import numpy as np

from datarray.datarray import (Axis, DataArray, NamedAxisError, AxesManager,
                               _pull_axis, _reordered_axes)

import nose.tools as nt
import numpy.testing as npt

DA = DataArray(np.random.randn(4, 2, 6), 'xyz')
YZ = AxesManager(DA, (Axis('y', 0, None), Axis('z', 1, None)))
XZ = AxesManager(DA, (Axis('x', 0, None), Axis('z', 1, None)))
XY = AxesManager(DA, (Axis('x', 0, None), Axis('y', 1, None)))
AXES_REMOVED = dict(x=YZ, y=XZ, z=XY)


def test_axis_equal():
    ax1 = Axis('aname', 0, None)
    ax2 = Axis('aname', 0, None)
    nt.assert_equal(ax1, ax2)
    # The array to which the axis points does not matter in comparison
    ax3 = Axis('aname', 0, np.arange(10))
    nt.assert_equal(ax1, ax3)
    # but the index does
    ax4 = Axis('aname', 1, None)
    nt.assert_not_equal(ax1, ax4)
    # so does the name
    ax5 = Axis('anothername', 0, None)
    nt.assert_not_equal(ax1, ax5)
    # and obviously both
    nt.assert_not_equal(ax4, ax5)
    # Try with labels
    ax6 = Axis('same', 0, None, labels=['a', 'b'])
    ax7 = Axis('same', 0, None, labels=['a', 'b'])
    nt.assert_equal(ax6, ax7)
    ax8 = Axis('same', 0, None, labels=['a', 'xx'])
    nt.assert_not_equal(ax6, ax8)

def test_bad_labels1():
    d = np.zeros(5)
    # bad labels length
    nt.assert_raises(ValueError, DataArray, d, axes=[('a', 'uvw')])

def test_bad_labels2():
    d = np.zeros(5)
    # uniqueness error
    nt.assert_raises(ValueError, DataArray, d, axes=[('a', ['u']*5)])

def test_bad_labels3():
    d = np.zeros(5)
    # type error
    nt.assert_raises(ValueError, DataArray, d, axes=[('a', [1, 1, 1, 1, 1])])
    
def test_basic():
    adata = [2,3]
    a = DataArray(adata, 'x', float)
    nt.assert_equal(a.names, ('x',))
    nt.assert_equal(a.dtype, np.dtype(float))
    b = DataArray([[1,2],[3,4],[5,6]], 'xy')
    nt.assert_equal(b.names, ('x','y'))
    # integer slicing
    b0 = b.axes.x[0]
    npt.assert_equal(b0, [1,2])
    # slice slicing
    b1 = b.axes.x[1:]
    npt.assert_equal(b1, [[3,4], [5,6]])

def test_bad_axes_axes():
    d = np.random.randn(3,2)
    nt.assert_raises(NamedAxisError, DataArray, d, axes='xx')

def test_combination():
    narr = DataArray(np.zeros((1,2,3)), axes=('a','b','c'))
    n3 = DataArray(np.ones((1,2,3)), axes=('x','b','c'))
    nt.assert_raises(NamedAxisError, np.add, narr, n3)
    # addition of scalar
    res = narr + 2
    nt.assert_true(isinstance(res, DataArray))
    nt.assert_equal(res.axes, narr.axes)
    # addition of matching size array, with matching names
    res = narr + narr
    nt.assert_equal(res.axes, narr.axes)

def test_label_change():
    a = DataArray([1,2,3])
    nt.assert_equal(a.names, (None,))
    a.axes[0].name = "test"
    nt.assert_equal(a.names, ("test",))

def test_1d():
    adata = [2,3]
    a = DataArray(adata, 'x', int)
    # Verify scalar extraction
    nt.assert_true(np.isscalar(a.axes.x[0]))
    nt.assert_equal(np.dtype(a.axes.x[0]), np.dtype(np.int))
    # Verify indexing of axis
    nt.assert_equals(a.axes.x.index, 0)
    # Iteration checks
    for i,val in enumerate(a.axes.x):
        nt.assert_equals(val, adata[i])
        nt.assert_true(np.isscalar(val))
        nt.assert_equal(np.dtype(val), np.dtype(np.int))

def test_2d():
    b = DataArray([[1,2],[3,4],[5,6]], 'xy')
    nt.assert_equals(b.names, ('x', 'y'))
    # Check row named slicing
    rs = b.axes.x[0]
    npt.assert_equal(rs, [1,2])
    nt.assert_equal(rs.names, ('y',))
    nt.assert_equal(tuple(rs.axes), (Axis('y', 0, rs),))
    # Now, check that when slicing a row, we get the right names in the output
    nt.assert_equal(b.axes.x[1:].names, ('x','y'))
    # Check column named slicing
    cs = b.axes.y[1]
    npt.assert_equal(cs, [2, 4, 6])
    nt.assert_equal(cs.names, ('x',))
    nt.assert_equal(tuple(cs.axes), (Axis('x', 0, cs),))
    # What happens if we do normal slicing?
    rs = b[0]
    npt.assert_equal(rs, [1, 2])
    nt.assert_equal(rs.names, ('y',))
    nt.assert_equal(tuple(rs.axes), (Axis('y', 0, rs),))

def test__pull_axis():
    a = Axis('x', 0, None)
    b = Axis('y', 1, None)
    c = Axis('z', 2, None)
    t_pos = Axis('y', 1, None)
    t_neg = Axis('x', 5, None)
    axes = [a, b, c]
    nt.assert_true(t_pos in axes)
    nt.assert_false(t_neg in axes)
    nt.assert_equal(axes, _pull_axis(axes, t_neg))
    nt.assert_equal(axes[:-1], _pull_axis(axes, c))
    new_axes = [a, Axis('z', 1, None)]
    nt.assert_equal(new_axes, _pull_axis(axes, t_pos))

def test__reordered_axes():
    a = Axis('x', 0, None)
    b = Axis('y', 1, None)
    c = Axis('z', 2, None)
    res = _reordered_axes([a,b,c], (1,2,0))
    names_inds = [(ax.name, ax.index) for ax in res]
    nt.assert_equal(set(names_inds), set([('y',0),('z',1),('x',2)]))

def test_axis_set_name():
    a = DataArray(np.arange(20).reshape(2,5,2), 'xyz')
    a.axes[0].set_name('u')
    nt.assert_equal(a.axes[0].name, 'u', 'name change failed')
    nt.assert_equal(a.axes.u, a.axes[0], 'name remapping failed')
    nt.assert_equal(a.axes.u.index, 0, 'name remapping failed')

def test_array_set_name():
    a = DataArray(np.arange(20).reshape(2,5,2), 'xyz')
    a.set_name(0, 'u')
    nt.assert_equal(a.axes[0].name, 'u', 'name change failed')
    nt.assert_equal(a.axes.u, a.axes[0], 'name remapping failed')
    nt.assert_equal(a.axes.u.index, 0, 'name remapping failed')
    
def test_axis_make_slice():
    p_arr = np.random.randn(2,4,5)
    ax_spec = 'capitals', ['washington', 'london', 'berlin', 'paris', 'moscow']
    d_arr = DataArray(p_arr, [None, None, ax_spec])
    a = d_arr.axes.capitals
    sl = a.make_slice( slice('london', 'moscow')  )
    should_be = ( slice(None), slice(None), slice(1,4) )
    nt.assert_equal(should_be, sl, 'slicing tuple from labels not correct')
    sl = a.make_slice( slice(1,4) )
    nt.assert_equal(should_be, sl, 'slicing tuple from idx not correct')

# also test with the slicing syntax
def test_labels_slicing():
    p_arr = np.random.randn(2,4,5)
    ax_spec = 'capitals', ['washington', 'london', 'berlin', 'paris', 'moscow']
    d_arr = DataArray(p_arr, [None, None, ax_spec])
    a = d_arr.axes.capitals
    sub_arr = d_arr.axes.capitals['washington'::2]
    nt.assert_equal(sub_arr.axes.capitals.labels,
                    a.labels[0::2])
    nt.assert_true((sub_arr == d_arr[:,:,0::2]).all())

# -- Tests for reshaping -----------------------------------------------------

def test_flatten_and_ravel():
    "Test the functionality of ravel() and flatten() methods"
    d = DataArray(np.arange(20).reshape(4,5), 'xy')
    df = d.flatten()
    nt.assert_true(type(df) is np.ndarray, 'Type error in flatten')
    nt.assert_true(df.shape == (20,), 'Wrong shape in flatten')
    df[:4] = 0
    nt.assert_false((d[0,:4] == 0).all(), 'Copy not made in flatten')

    dr = d.ravel()
    nt.assert_true(type(dr) is np.ndarray, 'Type error in ravel')
    nt.assert_true(dr.shape == (20,), 'Wrong shape in ravel')
    dr[:4] = 0
    nt.assert_true((d[0,:4] == 0).all(), 'View not made in ravel')

def test_squeeze():
    "Test squeeze method"
    d = DataArray(np.random.randn(3,2,9), 'xyz')
    d2 = d[None,:,None,:,:,None]
    nt.assert_true(d2.shape == (1,3,1,2,9,1), 'newaxis slicing failed')
    d3 = d.squeeze()
    nt.assert_true(d3.shape == d.shape,
                   'squeezing length-1 dimensions failed')
    nt.assert_true(d3.names == d.names, 'Axes got lost in squeeze')

def test_reshape():
    d = DataArray(np.random.randn(3,4,5), 'xyz')
    new_shape = (1,3,1,4,5)
    # Test padding the shape
    d2 = d.reshape(new_shape)
    new_labels = (None, 'x', None, 'y', 'z')
    nt.assert_true(d2.names == new_labels,
                   'Array with inserted dimensions has wrong labels')
    nt.assert_true(d2.shape == new_shape, 'New shape wrong')

    # Test trimming the shape
    d3 = d2.reshape(d.shape)
    nt.assert_true(d3.names == d.names,
                   'Array with removed dimensions has wrong labels')
    nt.assert_true(d3.shape == d.shape, 'New shape wrong')

    # Test a combo of padding and trimming
    d4 = d2.reshape(3,4,1,5,1)
    new_labels = ('x', 'y', None, 'z', None)
    nt.assert_true(
        d4.names == new_labels,
        'Array with inserted and removed dimensions has wrong labels')
    nt.assert_true(d4.shape == (3, 4, 1, 5, 1), 'New shape wrong')

def test_reshape_corners():
    "Test some corner cases for reshape"
    d = DataArray(np.random.randn(3,4,5), 'xyz')
    d2 = d.reshape(-1)
    nt.assert_true(d2.shape == (60,), 'Flattened shape wrong')
    nt.assert_true(type(d2) is np.ndarray, 'Flattened type wrong')

    d2 = d.reshape(60)
    nt.assert_true(d2.shape == (60,), 'Flattened shape wrong')
    nt.assert_true(type(d2) is np.ndarray, 'Flattened type wrong')
    
def test_axis_as_index():
    narr = DataArray(np.array([[1, 2, 3], [4, 5, 6]]), axes=('a', 'b'))
    npt.assert_array_equal(np.sum(narr, axis=narr.axes.a), [5, 7, 9])

# -- Tests for redefined methods ---------------------------------------------
    
def test_transpose():
    b = DataArray([[1,2],[3,4],[5,6]], 'xy')
    bt = b.T
    c = DataArray([ [1,3,5], [2,4,6] ], 'yx')
    nt.assert_true(bt.axes.x.index == 1 and bt.axes.y.index == 0)
    nt.assert_true(bt.shape == (2,3))
    nt.assert_true((bt==c).all())

def test_swapaxes():
    n_arr = np.random.randn(2,4,3)
    a = DataArray(n_arr, 'xyz')
    b = a.swapaxes('x', 'z')
    c = DataArray(n_arr.transpose(2,1,0), 'zyx')
    nt.assert_true((c==b).all(), 'data not equal in swapaxes test')
    for ax1, ax2 in zip(b.axes, c.axes):
        nt.assert_true(ax1==ax2, 'axes not equal in swapaxes test')

# -- Tests for wrapped ndarray methods ---------------------------------------

reductions = ['mean', 'var', 'std', 'min',
              'max', 'sum', 'prod', 'ptp', 'any', 'all',
              'argmax', 'argmin']
accumulations = ['cumprod', 'cumsum']

methods = reductions + accumulations

def check_data_axes(d_arr, op, axis, exp_axes, *args, **kwargs):
    """ Check data and axes correct after operation `op`
    """
    from datarray.datarray import _names_to_numbers
    super_opr = getattr(np.ndarray, op)
    axis_idx = _names_to_numbers(d_arr.axes, [axis])[0]
    d1 = super_opr(np.asarray(d_arr), axis_idx, *args, **kwargs)
    opr = getattr(d_arr, op)
    d_arr_out = opr(axis, *args, **kwargs)
    nt.assert_equal(d_arr_out.axes, exp_axes)
    d2 = np.asarray(d_arr_out)
    npt.assert_equal(d1.shape, d2.shape)
    npt.assert_array_equal(d1, d2)


def test_wrapped_ops_data():
    a = DataArray(np.random.randn(4,2,6), 'xyz')
    for m in methods:
        check_data_axes(a, m, 'x', YZ if m in reductions else DA.axes)
        check_data_axes(a, m, 'y', XZ if m in reductions else DA.axes)
        check_data_axes(a, m, 'z', XY if m in reductions else DA.axes)


def test_reductions_keepdims():
    names = 'xyz'
    a = np.arange(24).reshape((2, 3, 4))
    da = DataArray(a, names)
    for idx, name in enumerate(names):
        axes_removed = AXES_REMOVED[name]
        # Test keepdims as kwarg
        for method in reductions:
            check_data_axes(da, method, name, axes_removed)
            if method not in ('ptp', 'argmin', 'argmax'):
                # Reductions taking keepdims argument
                check_data_axes(da, method, name, DA.axes, keepdims=True)
        # Test the individual functions with positional args
        dt = np.dtype(float)
        out = np.mean(da, axis=name)
        kd_out = DataArray(np.mean(a, axis=idx, keepdims=True), names)
        # Functions with signature axis, dtype, out, keepdims
        for method in ('mean', 'sum', 'prod', 'all', 'any'):
            check_data_axes(da, method, name, axes_removed, dt, out)
            check_data_axes(da, method, name, DA.axes, dt, kd_out, True)
        # Signature axis, out, dtype, ddof, keepdims
        for method in ('var', 'std'):
            check_data_axes(da, method, name, axes_removed, dt, out, 0)
            check_data_axes(da, method, name, DA.axes, dt, kd_out, 0, True)
        # Signature axis, out, keepdims
        for method in ('min', 'max'):
            check_data_axes(da, method, name, axes_removed, out)
            check_data_axes(da, method, name, DA.axes, kd_out, True)
        # Test reductions not using keepdims
        out_int = out.astype(np.intp)  # argmin/max have integer output
        for method in ('argmin', 'argmax'):
            check_data_axes(da, method, name, axes_removed, out_int)
        check_data_axes(da, 'ptp', name, axes_removed, out)


# -- Tests for slicing with "newaxis" ----------------------------------------
def test_newaxis_slicing():
    b = DataArray([[1,2],[3,4],[5,6]], 'xy')
    b2 = b[np.newaxis]
    nt.assert_true(b2.shape == (1,) + b.shape)
    nt.assert_true(b2.axes[0].name == None)

    b2 = b[:,np.newaxis]
    nt.assert_true(b2.shape == (3,1,2))
    nt.assert_true((b2[:,0,:]==b).all())

# -- Testing broadcasting features -------------------------------------------
def test_broadcast():
    b = DataArray([[1,2],[3,4],[5,6]], 'xy')
    a = DataArray([1,0], 'y')
    # both of these should work
    c = b + a
    nt.assert_true(c.names == ('x', 'y'), 'simple broadcast failed')
    c = a + b
    nt.assert_true(c.names == ('x', 'y'),
                   'backwards simple broadcast failed')
    
    a = DataArray([1, 1, 1], 'x')
    # this should work too
    c = a[:,np.newaxis] + b
    nt.assert_true(c.names == ('x', 'y'), 'forward broadcast1 failed')
    c = b + a[:,np.newaxis] 
    nt.assert_true(c.names == ('x', 'y'), 'forward broadcast2 failed')

    b = DataArray(np.random.randn(3,2,4), ['x', None, 'y'])
    a = DataArray(np.random.randn(2,4), [None, 'y'])
    # this should work
    c = b + a
    nt.assert_true(c.names == ('x', None, 'y'),
                   'broadcast with unlabeled dimensions failed')
    # and this
    a = DataArray(np.random.randn(2,1), [None, 'y'])
    c = b + a
    nt.assert_true(
        c.names == ('x', None, 'y'),
        'broadcast with matched name, but singleton dimension failed')
    # check that labeled Axis names the resulting Axis
    b = DataArray(np.random.randn(3,2,4), ['x', 'z', 'y'])
    a = DataArray(np.random.randn(2,4), [None, 'y'])
    # this should work
    c = b + a
    nt.assert_true(c.names == ('x', 'z', 'y'),
                   'broadcast with unlabeled dimensions failed')


# -- Testing slicing failures ------------------------------------------------
@nt.raises(NamedAxisError)
def test_broadcast_fails1():
    a = DataArray( np.random.randn(5,6), 'yz' )
    b = DataArray( np.random.randn(5,6), 'xz' )
    c = a + b

@nt.raises(ValueError)
def test_broadcast_fails2():
    a = DataArray( np.random.randn(2,5,6), 'xy' ) # last axis is unlabeled
    b = DataArray( np.random.randn(2,6,6), 'xy' )
    # this should fail simply because the dimensions are not matched
    c = a + b

@nt.raises(IndexError)
def test_indexing_fails():
    "Ensure slicing non-existent dimension fails"
    a = DataArray( np.random.randn(2,5,6), 'xy' )
    a[:2,:1,:2,:5]

@nt.raises(IndexError)
def test_ambiguous_ellipsis_fails():
    a = DataArray( np.random.randn(2,5,6), 'xy' )
    a[...,0,...]

def test_ellipsis_slicing():
    a = DataArray( np.random.randn(2,5,6), 'xy' )
    nt.assert_true((a[...,0] == a[:,:,0]).all(),
                   'slicing with ellipsis failed')
    nt.assert_true((a[0,...] == a[0]).all(),
                   'slicing with ellipsis failed')
    nt.assert_true((a[0,...,0] == a[0,:,0]).all(),
                   'slicing with ellipsis failed')

def test_shifty_axes():
    arr = np.random.randn(2,5,6)
    a = DataArray( arr, 'xy' )
    # slicing out the "x" Axis triggered the unlabeled axis to change
    # name from "_2" to "_1".. make sure that this change is mapped
    b = a[0,:2]
    nt.assert_true((b == arr[0,:2]).all(), 'shifty axes strike again!')
    
# -- Testing utility functions -----------------------------------------------
from datarray.datarray import _expand_ellipsis, _make_singleton_axes

def test_ellipsis_expansion():
    slicing = ( slice(2), Ellipsis, 2 )
    fixed = _expand_ellipsis(slicing, 4)
    should_be = ( slice(2), slice(None), slice(None), 2 )
    nt.assert_true(fixed==should_be, 'wrong slicer1')
    fixed = _expand_ellipsis(slicing, 2)
    should_be = ( slice(2), 2 )
    nt.assert_true(fixed==should_be, 'wrong slicer2')

def test_singleton_axis_prep():
    b = DataArray( np.random.randn(5,6), 'xz' )
    slicing = ( None, )
    shape, axes, key = _make_singleton_axes(b, slicing)

    key_should_be = (slice(None), ) # should be trimmed
    shape_should_be = (1,5,6)
    ax_should_be = [ Axis(l, i, b) for i, l in enumerate((None, 'x', 'z')) ]

    nt.assert_true(key_should_be==key, 'key translated poorly')
    nt.assert_true(shape_should_be==shape, 'shape computed poorly')
    nt.assert_true(all([a1==a2 for a1,a2 in zip(ax_should_be, axes)]),
                   'axes computed poorly')

def test_singleton_axis_prep2():
    # a little more complicated
    b = DataArray( np.random.randn(5,6), 'xz' )
    slicing = ( 0, None )
    shape, axes, key = _make_singleton_axes(b, slicing)

    key_should_be = (0, ) # should be trimmed
    shape_should_be = (5,1,6)
    ax_should_be = [ Axis(l, i, b) for i, l in enumerate(('x', None, 'z')) ]

    nt.assert_true(key_should_be==key, 'key translated poorly')
    nt.assert_true(shape_should_be==shape, 'shape computed poorly')
    nt.assert_true(all([a1==a2 for a1,a2 in zip(ax_should_be, axes)]),
                   'axes computed poorly')
    
# -- Test binary operations --------------------------------------------------

def test_label_mismatch():
    dar1 = DataArray([1, 2], [('time', ['A1', 'B1'])])
    dar2 = DataArray([1, 2], [('time', ['A2', 'B2'])])
    nt.assert_raises(NamedAxisError, dar1.__add__, dar2)
    nt.assert_raises(NamedAxisError, dar1.__sub__, dar2)
    nt.assert_raises(NamedAxisError, dar1.__mul__, dar2)
    nt.assert_raises(NamedAxisError, dar1.__floordiv__, dar2)
    nt.assert_raises(NamedAxisError, dar1.__truediv__, dar2)
    if not PY3:
        nt.assert_raises(NamedAxisError, dar1.__div__, dar2)
    
# -- Test DataArray.axes
class TestAxesManager(object):
    def setUp(self):
        self.axes_spec = ('date', ('stocks', ('aapl', 'ibm', 'goog', 'msft')), 'metric')
        self.A = DataArray(np.random.randn(200, 4, 10), axes=self.axes_spec)

    def test_axes_name_collision(self):
        "Test .axes object for attribute collisions with axis names"
        A = DataArray(np.arange(6).reshape([1,2,3]), 
                ('_arr', '_axes', '_namemap'))
        nt.assert_true(A.axes[0] is A.axes('_arr') is A.axes._arr)
        nt.assert_true(A.axes[1] is A.axes('_axes') is A.axes._axes)
        nt.assert_true(A.axes[2] is A.axes('_namemap') is A.axes._namemap)
        
        # Try to invoke some methods that use these attributes internally
        B = A[np.newaxis, ...]
        nt.assert_equal(B.shape, (1,1,2,3))
        nt.assert_true(np.all(A + A == 2*A))

    def test_axes_numeric_access(self):
        for i,spec in enumerate(self.axes_spec):
            try:
                name,labels = spec
            except ValueError:
                name,labels = spec,None
            nt.assert_true(self.A.axes[i] == Axis(name=name, index=i,
                parent_arr=self.A, labels=labels))

    def test_axes_attribute_access(self):
        for spec in self.axes_spec:
            try:
                name,labels = spec
            except ValueError:
                name,labels = spec,None
            nt.assert_true(getattr(self.A.axes, name) is self.A.axes(name))

    def test_equality(self):
        B = DataArray(np.random.randn(200, 4, 10), axes=self.axes_spec)
        nt.assert_true(self.A.axes == B.axes)
        # What if axes differ by labels only?
        D = DataArray(np.random.randn(200, 4, 10), axes=('date', 'stocks', 'metric')) 
        nt.assert_false(self.A.axes == D.axes)
