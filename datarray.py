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

Basic array creation and axis access:

>>> narr = DataArray(np.zeros((1,2,3)), names=('a','b','c'))
>>> narr.axis.a
Axis 'a': index 0, length 1
>>> narr.axis.b
Axis 'b': index 1, length 2
>>> narr.axis.c
Axis 'c': index 2, length 3
>>> narr.shape
(1, 2, 3)

Combining named and unnamed arrays:
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

ToDo
====

- Implementing axes with values in them (a la Per Sederberg)

- Support DataArray instances with mixed axes: simple ones with no values and
'fancy' ones with data in them.  Syntax?

DataArray.from_names(data, names=['a','b','c'])

DataArray(data, axes=[('a',[1,2,3]), ('b',['one','two']),
('c',['red','black'])])

DataArray(data, axes=[('a',[1,2,3]), ('b',None), ('c',['red','black'])])

- We need to support unnamed axes.

- Units support (Darren's)

- Jagged arrays? Kilian's suggestion.  Drop the base array altogether, and
access data via the .axis objects alone.

- "Enum dtype", could be useful for event selection.

- "Ordered factors"? Something R supports.


- How many axis classes?


Axis api: if a is an axis from an array: a = x.axis.a

a.at(key): return the slice at that key, with one less dimension than x
a.keep(keys): join slices for given keys, dims=dims(x)
a.drop(keys): like keep, but the opposite

a[i] valid cases:
i: integer => normal numpy scalar indexing, one less dim than x
i: slice: numpy view slicing.  same dims as x, must recover the ticks 
i: list/array: numpy fancy indexing, as long as the index list is 1d only.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import copy

import numpy as np

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

class NamedAxisError(Exception):
    pass


class KeyStruct(object):
    """A slightly enhanced version of a struct-like class with named key access.

    Examples
    --------
    
    >>> a = KeyStruct()
    >>> a.x = 1
    >>> a['x']
    1
    >>> a['y'] = 2
    >>> a.y
    2
    >>> a[3] = 3
    Traceback (most recent call last):
      ... 
    TypeError: attribute name must be string, not 'int'

    >>> b = KeyStruct(x=1, y=2)
    >>> b.x
    1
    >>> b['y']
    2
    """
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        setattr(self, key, val)


class Axis(object):
    """Object to access a given axis of an array.

    Key point: every axis contains a  reference to its parent array!
    """
    def __init__(self, name, index, parent_arr, ticks=None):
        self.name = name
        self.index = index
        self.parent_arr = parent_arr
        # This will raise if the ticks are invalid:
        self._tick_dict = self._validate_ticks(ticks)
        self.ticks = ticks

    def _validate_ticks(self, ticks):
        """Validate constraints on ticks.

        Ensure:

        - uniqueness
        - length
        """
        if ticks is None:
            return None
        # We always store ticks as numpy arrays
        #ticks = np.asarray(ticks)

        nticks = len(ticks)
        # Sanity check: the first dimension must match that of the parent array
        if nticks != self.parent_arr.shape[self.index]:
            e = "Dimension mismatch between ticks and data at index %i" % \
                self.index
            raise ValueError(e)
        
        # Validate uniqueness
        t_dict = dict(zip(ticks, range(nticks)))
        if len(t_dict) != nticks:
            raise ValueError("non-unique tick values not supported")
        return t_dict
        

    def __len__(self):
        return self.parent_arr.shape[self.index]

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

    def __str__(self):
        return 'Axis %r: index %i, length %i' % \
               (self.name, self.index, len(self))

    __repr__ = __str__
    
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
        parent_arr = self.parent_arr # local for speed
        parent_arr_ndim = parent_arr.ndim
        # The logic is: when using scalar indexing, the dimensionality of the
        # output is parent_arr.ndim-1, while when using slicing the output has
        # the same number of dimensions as the input.  For this reason, the
        # case when parent_arr.ndim is 1 and the indexing is scalar needs to be
        # handled separately, since the output will be 0-dimensional.  In that
        # case, we must return the plain scalar and not build a slice object
        # that would return a 1-element sub-array.
        #
        # XXX we do not here handle 0 dimensional arrays.
        # XXX fancy indexing
        if parent_arr_ndim == 1 and not isinstance(key, slice):
            return parent_arr[key]
        # XXX TODO - fancy indexing
        # For other cases (slicing or scalar indexing of ndim>1 arrays),
        # build the proper slicing object to cut into the managed array
        fullslice = [slice(None)] * parent_arr_ndim
        fullslice[self.index] = key
        #print 'getting output'  # dbg
        out = parent_arr[fullslice]
        #print 'returning output'  # dbg
        if out.ndim != parent_arr_ndim:
            # We lost a dimension, drop the axis!
            _set_axes(out, _pull_axis(parent_arr.axes, self))
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
    def __init__(self, index, parent_arr):
        # XXX use super here?
        Axis.__init__(self,'unnamed_%s' % index,index, parent_arr)


def _names_to_numbers(axes, ax_ids):
    ''' Convert any axis names to axis indices '''
    proc_ids = []
    for ax_id in ax_ids:
        if isinstance(ax_id, basestring):
            matches = [ax for ax in axes if ax.name == ax_id]
            if not matches:
                raise NamedAxisError('No axis named %s' % ax_id)
            proc_ids.append(matches[0].index)
        else:
            proc_ids.append(ax_id)
    return proc_ids


def _pull_axis(axes, target_axis):
    ''' Return axes removing any axis matching `target_axis`'''
    axes = axes[:]
    try:
        # XXX - what is this? remove returns None!  And ind isn't used below
        ind = axes.remove(target_axis)
    except ValueError:
        return axes
    rm_i = target_axis.index
    for i, ax in enumerate(axes):
        if ax.index >=rm_i:
            axes[i] = ax.__class__(ax.name, ax.index-1, ax.parent_arr)
    return axes


def _set_axes(dest, in_axes):
    """Set the axes in `dest` from `in_axes`.

    WARNING: The destination is modified in-place!  The following attributes
    are added to it:

    - axis: a KeyStruct with each axis as a named attribute.
    - axes: a list of all axis instances.
    - names: a list of all the axis names.

    Parameters
    ----------
      dest : array
      in_axes : sequence of axis objects
    """
    # XXX here is where the logic is implemented for missing names.
    # Here there are no named axis objects if there are fewer names than
    # axes in the array
    axes = []
    names = []
    ax_holder = KeyStruct()
    for ax in in_axes:
        new_ax = ax.__class__(ax.name, ax.index, dest)
        axes.append(new_ax)
        names.append(ax.name)
        ax_holder[ax.name] = new_ax
    dest.axes = axes
    dest.names = names
    dest.axis = ax_holder
    

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
           None if triggered from DataArray.__new__ call
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
        """ accept integer or named axes, reorder axes """
        # implement tuple-or-*args logic of np.transpose
        axes = list(axes)
        if not axes:
            axes = range(self.ndim)
        # expand sequence if sequence passed as first and only arg
        elif len(axes) == 1:
            try:
                axes = axes[0][:]
            except TypeError:
                pass
        proc_axids = _names_to_numbers(self.axes, axes)
        out = self.transpose(proc_axids)
        _set_axes(out, _reordered_axes(self.axes, proc_axids))
        return out


def _reordered_axes(axes, axis_indices):
    ''' Perform axis reordering according to `axis_indices`

    Parameters
    ----------
    axes : sequence of axes
       The axis indices in this list reflect the axis ordering before
       the permutation given by `axis_indices`
    axis_indices : sequence of ints
       indices giving new order of axis numbers

    Returns
    -------
    ro_axes : sequence of axes
       sequence of axes in arbitrary order with axis indices reflecting
       reordering given by `axis_indices`

    Examples
    --------
    >>> a = Axis('x', 0, None)
    >>> b = Axis('y', 1, None)
    >>> c = Axis('z', 2, None)
    >>> res = _reordered_axes([a,b,c], (1,2,0))
    '''
    for new_ind, old_ind in enumerate(axis_indices):
        pass
    #raise NotImplementedError
