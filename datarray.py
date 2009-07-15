"""Data arrays...

Questions

- How to handle broadcasting? Use UnnamedAxis OK?

- Slicing
- Broadcasting
- Transposition
- Swapaxes
- Rollaxes
- Wrapping functions with 'axis=' kw.

"""

import copy

import numpy as np


### classes

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
        # Make a slice
        nullslice = slice(None)
        fullslice = [nullslice] * self.arr.ndim
        fullslice[self.index] = key
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


class DataArray(np.ndarray):
    @property
    def T(self):
        return self.transpose()
        
    def __new__(cls, data, names=None, dtype=None, copy=False):
        # Ensure the output is an array of the proper type
        arr = np.array(data, dtype=dtype, copy=copy)
        arr = arr.view(cls)

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
        # provide info for what's happening
        print "finalize:\t%s\n\t\t%s" % (self.__class__, obj.__class__) # dbg
        # provide more info
        if hasattr(obj,'names'):
            copy_names(obj,self)

    def transpose(self, *axes):
        print 'hi'


### main
dt = np.dtype([('f1',float), ('f2',float)])

a = DataArray([2,3], 'x', int)
b = DataArray([[1,2],[3,4]], 'xy')

print b.a_y[1].names
