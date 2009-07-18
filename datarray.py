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
        arr = self.arr # local for speed
        arr_ndim = arr.ndim
        # The logic is: when using scalar indexing, the dimensionality of the
        # output is arr.ndim-1, while when using slicing the output has
        # the same number of dimensions as the input.  For this reason, the
        # case when arr.ndim is 1 and the indexing is scalar needs to be
        # handled separately, since the output will be 0-dimensional.  In that
        # case, we must return the plain scalar and not build a slice object
        # that would return a 1-element sub-array.
        if arr_ndim == 1 and not isinstance(key, slice):
            return arr[key]

        # For other cases (slicing or scalar indexing of ndim>1 arrays), build
        # the proper slicing object to cut into the managed array
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
    """A class to tag unnamed axes"""
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
        if names is not None and len(names) > arr.ndim:
            raise ValueError("names list longer than array ndim")

        # Set the given names
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
    
