"""Data arrays...

Questions

- How to handle broadcasting? Use UnnamedAxis OK?

- Slicing
- Broadcasting
- Transposition
- Swapaxes
- Rollaxes
- Iteration
- Wrapping functions with 'axis=' kw.

Combining named and unnamed arrays:

narr = DataArray(np.zeros((1,2,3), names=('a','b','c'))
res = narr + 5 # OK
res = narr + np.zeros((1,2,3)) # OK
n2 = DataArray(np.ones((1,2,3), names=('a','b','c'))
res = narr + n2 # OK
n3 = DataArray(np.ones((1,2,3), names=('x','b','c'))
res = narr + n3 # Raises NamedAxisError

Now, what about matching names, but different indices for the names?

n4 = DataArray(np.ones((2,1,3), names=('b','a','c'))
res = narr + n4 # is this OK?
res.shape

Maybe this is too much magic?  Probably the names and the position has
to be the same, and the above example should raise an error.  At least
for now we will raise an error, and review later.

What about broadcasting between two named arrays, where the broadcasting
adds an axis?

a = DataArray(np.zeros((3,), names=('a',))
b = DataArray(np.zeros((2,3), names=('a','b'))
res = a + b
res.names == ('a', 'b')
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import copy

import nose.tools as nt
import numpy as np
import numpy.testing as npt

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

class Axis(object):
    """Object to access a given axis of an array.

    Key point: every axis contains a  reference to its parent array!
    """
    def __init__(self, name, index, arr):
        self.name = name
        self.index = index
        self.arr = arr

    def __len__(self):
        return self.arr.shape[self.index]
    
    def __getitem__(self, key):
        # `key` can be one of:
        # * integer (more generally, any valid scalar index)
        # * slice
        # * list (fancy indexing)
        # * array (fancy indexing)
        #
        # XXX We don't handle fancy indexing at the moment
        # if isinstance(key, (np.ndarray, list)):
        #    raise TypeError('We do not handle fancy indexing yet')
        # If there is a change in dimensionality of the result, the
        # answer will have to be a normal array
        # If the dimensionality is preserved, we can keep the structure
        # of the parent
        arr = self.arr # local for speed
        arr_ndim = arr.ndim
        # The logic is: when using scalar indexing, the dimensionality of the
        # output is arr.ndim-1, while when using slicing the output has
        # the same number of dimensions as the input.  For this reason, the
        # case when arr.ndim is 1 and the indexing is scalar needs to be
        # handled separately, since the output will be 0-dimensional.  In that
        # case, we must return the plain scalar and not build a slice object
        # that would return a 1-element sub-array.
        #
        # XXX we do not here handle 0 dimensional arrays.
        # XXX fancy indexing
        if arr_ndim == 1 and not isinstance(key, slice):
            return arr[key]
        # XXX Fancy indexing
        # For other cases (slicing or scalar indexing of ndim>1 arrays),
        # build the proper slicing object to cut into the managed array
        fullslice = [slice(None)] * arr_ndim
        fullslice[self.index] = key

        #print 'getting output'  # dbg
        out = arr[fullslice]
        #print 'returning output'  # dbg

        if out.ndim != arr_ndim:
            # We lost a dimension, drop the axis!
            kept_names = []
            for i,aname in enumerate(out.names):
                a_name = 'a_%s' % aname
                axis = getattr(out,a_name)
                if i==key:
                    #print "Dropping axis:",a_name  # dbg
                    delattr(out,a_name)
                else:
                    kept_names.append(aname)
            out.names = kept_names
        
        return out
        

class UnnamedAxis(Axis):
    """A class to tag unnamed axes

    Consider this case:
    narr = DataArray(np.zeros((1,2,3), names=('a',))

    in that case axes 1,2 do not have names.

    Options may be:
    * narr.names == ('a', 'unnamed_0', 'unnamed_1')
    * narr.names == ('a',)
    * narr.names == ('a', None, None)
    * narr.names == ('a', UnnamedAxis, UnnamedAxis)

    Then:

    narrt = narr.transpose()
    narrt.shape == (3,2,1)
    Now options may be:
    * narr.names == ('unamed_1', 'unnamed_0', 'a')
    * narr.names == ('a',)
    * narr.names == (None, None, 'a')
    * narr.names == (UnnamedAxis, UnnamedAxis, 'a')

    Consider broadcasting:

    narr = DataArray(np.zeros((3,), names=('a',))
    res = narr + np.ones((5,3))
    res.names == ?
    """
    def __init__(self, index, arr):
        # XXX use super here?
        Axis.__init__(self,'unnamed_%s' % index,index, arr)


def copy_names(src,dest):
    dest.names = src.names
    for i,name in enumerate(dest.names):
        aname = 'a_%s' % name
        newaxis = Axis(aname,i,dest)
        setattr(dest, aname, newaxis)


def names2namedict(names):
    """Make a name map out of any name input.
    """
    raise NotImplementedError() 


class DataArray(np.ndarray):

    # XXX- we need to figure out where in the numpy C code .T is defined!
    @property
    def T(self):
        return self.transpose()

    def __new__(cls, data, names=None, dtype=None, copy=False):
        # Ensure the output is an array of the proper type
        arr = np.array(data, dtype=dtype, copy=copy).view(cls)

        # Sanity check: if names are given, it must be a sequence no  longer
        # than the array shape
        # XXX check for len(names) == data.ndim ?
        if names is not None and len(names) > arr.ndim:
            raise ValueError("names list longer than array ndim")

        # Set the given names
        # XXX what happens if there are no names
        if names is not None:
            names = list(names)
            for i,name in enumerate(names):
                setattr(arr,'a_%s' % name,Axis(name, i, arr))
            arr.names = names

        # Or if the input had named axes, copy them over
        elif hasattr(data,'names'):
            copy_names(data, arr)
            
        return arr

    def __array_finalize__(self, obj):
        """Called by ndarray on subobject (like views/slices) creation.

        self: new object just made.
        obj: old object from which self was made.
        """
        
        #print "finalizing DataArray" # dbg
        
        # Ref: see http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        
        # provide info for what's happening
        #print "finalize:\t%s\n\t\t%s" % (self.__class__, obj.__class__) # dbg
        # provide more info
        if hasattr(obj,'names'):
            copy_names(obj,self)

    def transpose(self, *axes):
        raise NotImplementedError()


#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------
if 1:
    adata = [2,3]
    a = DataArray(adata, 'x', int)
    b = DataArray([[1,2],[3,4],[5,6]], 'xy')
    b0 = b.a_x[0]
    b1 = b.a_x[1:]
    
def test_1d():

    adata = [2,3]
    a = DataArray(adata, 'x', int)

    # Verify scalar extraction
    yield (nt.assert_true,isinstance(a.a_x[0],int))

    # Verify indexing of axis
    yield (nt.assert_equals, a.a_x.index, 0)

    # Iteration checks
    for i,val in enumerate(a.a_x):
        yield (nt.assert_equals,val,adata[i])
        yield (nt.assert_true,isinstance(val,int))


def test_2d():
    b = DataArray([[1,2],[3,4],[5,6]], 'xy')
    yield (nt.assert_equals, b.names, ['x','y'])

    # Check row slicing
    yield (npt.assert_equal, b.a_x[0], [1,2])

    # Check column slicing
    yield (npt.assert_equal, b.a_y[1], [2,4,6])

    # Now, check that when slicing a row, we get the right names in the output
    yield (nt.assert_equal, b.a_x[1:].names, ['x','y'])
    yield (nt.assert_equal, b.a_x[0].names, ['y'])
    
