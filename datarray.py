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
            kept_names = []
            for i,aname in enumerate(out.names):
                a_name = ax_attr_prefix + '%s' % aname
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


def copy_axes(src, dest):
    """ Copy arrays from DataArray `src` to array `dest`

    Creat axis objects to go with the names.
    Overwrite the names and axis objects (if any) from `dest`.

    Assumes that the Axes in `src` do in fact match the needed axes in
    `dest` - that is, that they have the correct indices, and there are
    the correct number of axes.
    """
    dest.names = src.names[:]
    axes = []
    for ax in src.axes:
        new_ax = ax.__class__(ax.name, ax.index, dest)
        axes.append(new_ax)
        setattr(dest, ax_attr_prefix + '%s' % ax.name, new_ax)
    dest.axes = axes


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
        axes = []
        if names is not None:
            names = list(names)
            for i,name in enumerate(names):
                ax = Axis(name, i, arr)
                setattr(arr, ax_attr_prefix + '%s' % name, ax)
                axes.append(ax)
            arr.names = names
            arr.axes = axes
        # Or if the input had named axes, copy them over
        elif hasattr(data,'axes'):
            copy_axes(data, arr)
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
        if not isinstance(obj, DataArray): # looks like view cast
            # XXX - here we have to decide what to do about axes without
            # names - unnamed axis?  None?  empty?
            self.axes = []
            self.names = []
            return
        # new-from-template: we need to know what's been done here. For
        # the moment, we'll assume no reducing operations or axis
        # changing manipulations have occurred.
        if self.shape != obj.shape:
            print 'Shapes do not match', self.shape, obj.shape
        copy_axes(obj, self)
            
    def transpose(self, *axes):
        raise NotImplementedError()


