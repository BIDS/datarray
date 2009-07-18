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
        if isinstance(key, slice):
            # Dimensionality of output is dimensionality of input
            pass
        else:
            # Scalar indexing, output has 1-d less
            if self.arr.ndim==1:
                return self.arr[key]
        # Make a slice
        nullslice = slice(None)
        fullslice = [nullslice] * self.arr.ndim
        fullslice[self.index] = key
        print 'FS:',fullslice
        return self.arr[fullslice]

        

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
        # Ref: see http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        
        # provide info for what's happening
        #print "finalize:\t%s\n\t\t%s" % (self.__class__, obj.__class__) # dbg
        # provide more info
        if hasattr(obj,'names'):
            copy_names(obj,self)


    def transpose(self, *axes):
        print 'hi'


#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------
if 1:
    adata = [2,3]
    a = DataArray(adata, 'x', int)
    print a.a_x[0]
    
def test1():

    adata = [2,3]
    a = DataArray(adata, 'x', int)

    yield (nt.assert_true,isinstance(a.a_x[0],int))
    
    yield (nt.assert_equals, a.a_x.index, 0)
    for i,val in enumerate(a.a_x):
        yield (nt.assert_equals,val,adata[i])
        yield (nt.assert_true,isinstance(val,int))

    b = DataArray([[1,2],[3,4]], 'xy')
    yield (nt.assert_equals, b.names, ['x','y'])

