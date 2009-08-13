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

>>> narr = DataArray(np.zeros((1,2,3)), names=('a','b','c'))
>>> res = narr + 5 # OK
>>> res = narr + np.zeros((1,2,3)) # OK
>>> n2 = DataArray(np.ones((1,2,3)), names=('a','b','c'))
>>> res = narr + n2 # OK

>>> n3 = DataArray(np.ones((1,2,3)), names=('x','b','c'))

res = narr + n3 # raises error
(NamedAxisError should be raised)
  ...

Now, what about matching names, but different indices for the names?

n4 = DataArray(np.ones((2,1,3)), names=('b','a','c'))
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

import numpy as np

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

ax_attr_prefix = 'ax_'

class NamedAxisError(Exception):
    pass


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

    def __eq__(self, other):
        ''' Axes are equal iff they have matching names and indices

        Parameters
        ----------
        other : ``Axis`` object
           Object to compare

        Returns
        -------
        tf : bool
           True if self == other

        Examples
        --------
        >>> ax = Axis('x', 0, np.arange(10))
        >>> ax == Axis('x', 0, np.arange(5))
        True
        >>> ax == Axis('x', 1, np.arange(10))
        False
        '''
        return self.name == other.name and self.index == other.index
    
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
        # XXX TODO - fancy indexing
        # For other cases (slicing or scalar indexing of ndim>1 arrays),
        # build the proper slicing object to cut into the managed array
        fullslice = [slice(None)] * arr_ndim
        fullslice[self.index] = key
        #print 'getting output'  # dbg
        out = arr[fullslice]
        #print 'returning output'  # dbg
        if out.ndim != arr_ndim:
            # We lost a dimension, drop the axis!
            _set_axes(out, _pull_axis(arr.axes, self))
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


def _pull_axis(axes, target_axis):
    ''' Return axes removing any axis matching `target_axis`'''
    axes = axes[:]
    try:
        ind = axes.remove(target_axis)
    except ValueError:
        return axes
    rm_i = target_axis.index
    for i, ax in enumerate(axes):
        if ax.index >=rm_i:
            axes[i] = ax.__class__(ax.name, ax.index-1, ax.arr)
    return axes


def _set_axes(dest, in_axes):
    # XXX here is where the logic is implemented for missing names.
    # Here there are no named axis objects if there are fewer names than
    # axes in the array
    axes = []
    names = []
    for ax in in_axes:
        new_ax = ax.__class__(ax.name, ax.index, dest)
        axes.append(new_ax)
        names.append(ax.name)
        setattr(dest, ax_attr_prefix + '%s' % ax.name, new_ax)
    dest.axes = axes
    dest.names = names
    

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
        if names is None:
            if hasattr(data,'axes'):
                _set_axes(arr, data.axes)
                return arr
            names = []
        elif len(names) > arr.ndim:
            raise NamedAxisError("names list longer than array ndim")
        axes = [Axis(name, i, arr) for i, name in enumerate(names)]
        _set_axes(arr, axes)
        return arr

    def __array_finalize__(self, obj):
        """Called by ndarray on subobject (like views/slices) creation.

        Parameters
        ----------
        self : ``DataArray``
           Newly create instance of ``DataArray``
        obj : ndarray or None
           any ndarray object (if view casting)
           ``DataArray`` instance, if new-from-template
           None if from DataArray(*args, **kwargs) call
        """
        
        #print "finalizing DataArray" # dbg
        
        # Ref: see http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        
        # provide info for what's happening
        #print "finalize:\t%s\n\t\t%s" % (self.__class__, obj.__class__) # dbg
        # provide more info
        if obj is None: # own constructor, we're done
            return
        if not hasattr(obj, 'axes'): # looks like view cast
            _set_axes(self, [])
            return
        # new-from-template: we just copy the names from the template,
        # and hope the calling rountine knows what to do with the output
        _set_axes(self, obj.axes)
            
    def transpose(self, *axes):
        raise NotImplementedError()


