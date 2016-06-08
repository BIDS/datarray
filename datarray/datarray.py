#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Py2-backwards compatibility
try:
  basestring
except NameError:
  basestring = str


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
    TypeError: hasattr(): attribute name must be string

    >>> b = KeyStruct(x=1, y=2)
    >>> b.x
    1
    >>> b['y']
    2
    >>> b['y'] = 4
    Traceback (most recent call last):
      ...
    AttributeError: KeyStruct already has atribute 'y'

    """
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        if hasattr(self, key):
            raise AttributeError('KeyStruct already has atribute %s'%repr(key))
        self.__dict__[key] = val

    def __setattr__(self, key, val):
        self[key] = val

class AxesManager(object):
    """
    Class to manage the logic of the datarray.axes object.
    
    >>> A = DataArray(np.random.randn(200, 4, 10), \
                axes=('date', ('stocks', ('aapl', 'ibm', 'goog', 'msft')), 'metric'))
    >>> isinstance(A.axes, AxesManager)
    True

    At a basic level, AxesManager acts like a sequence of axes:

    >>> A.axes # doctest:+ELLIPSIS
    (Axis(name='date', index=0, labels=None), ..., Axis(name='metric', index=2, labels=None))
    >>> A.axes[0]
    Axis(name='date', index=0, labels=None)
    >>> len(A.axes)
    3
    >>> A.axes[4]
    Traceback (most recent call last):
        ...
    IndexError: Requested axis 4 out of bounds
    
    Each axis is accessible as a named attribute:

    >>> A.axes.stocks
    Axis(name='stocks', index=1, labels=('aapl', 'ibm', 'goog', 'msft'))

    An axis can be indexed by integers or ticks:

    >>> np.all(A.axes.stocks['aapl':'goog'] == A.axes.stocks[0:2])
    True

    >>> np.all(A.axes.stocks[0:2] == A[:,0:2,:])
    True

    Axes can also be accessed numerically:

    >>> A.axes[1] is A.axes.stocks
    True

    Calling the AxesManager with string arguments will return an
    :py:class:`AxisIndexer` object which can be used to restrict slices to
    specified axes:

    >>> Ai = A.axes('stocks', 'date')
    >>> np.all(Ai['aapl':'goog', 100] == A[100, 0:2])
    True

    You can also mix axis names and integers when calling AxesManager.
    (Not yet supported.)

    # >>> np.all(A.axes(1, 'date')['aapl':'goog',100:200] == A[100:200, 0:2])
    # True
    """

    # The methods of this class use object.__getattribute__ to avoid a
    # potential collision between axis names and the internal instance
    # variables
    def __init__(self, arr, axes):
        self._arr = arr
        self._axes = tuple(axes)
        self._namemap = dict((ax.name,i) for i,ax in enumerate(axes))
    
    # This implements darray.axes.an_axis_name
    def __getattribute__(self, name):
        namemap = object.__getattribute__(self, '_namemap')
        axes = object.__getattribute__(self, '_axes')
        try:
            return axes[namemap[name]]
        except KeyError:
            return object.__getattribute__(self, name)

    def __len__(self):
        return len(object.__getattribute__(self, '_axes'))

    def __repr__(self):
        return str(tuple(self))

    def __getitem__(self, n):
        """Return the axis object at integer index `n`

        Parameters
        ----------
        n : int
            Index of axis to be returned.

        Returns
        -------
        ax : :class:`Axis` instance
            The requested :py:class:`Axis`.

        Examples
        --------
        >>> A = DataArray([[1,2],[3,4]], 'ab'); A.axes[0] is A.axes.a
        True
        >>> A.axes[1] is A.axes.b
        True
        """
        if not isinstance(n, int):
            raise TypeError("AxesManager expects integer index")
        try:
            return object.__getattribute__(self, '_axes')[n]
        except IndexError:
            raise IndexError("Requested axis %i out of bounds" % n)

    def __eq__(self, other):
        """Test for equality between two axes managers. Two axes managers are
        equal if the axes they manage are equal and have the same order.

        Examples
        --------
        >>> A = DataArray([[1,2],[3,4]], 'ab')
        >>> B = DataArray([[7,8],[9,10]], 'ab')
        >>> C = DataArray([[7,8],[9,10]], 'cd')
        >>> D = DataArray([[1,2,3,4],[5,6,7,8]], 'ab')
        >>> A.axes == B.axes
        True
        >>> A.axes == C.axes
        False
        >>> A.axes == D.axes
        True

        Parameters
        ----------
        other : any
    
        Returns
        -------
        out : bool

        """
        if not isinstance(other, AxesManager):
            return False
        axes = object.__getattribute__(self, '_axes')
        return axes == other._axes

    def __call__(self, *args):
        """Return an axis indexer object based on the supplied arguments.

        Parameters
        ----------
        args : sequence of strs
            A sequence of axis names.

        Returns
        -------
        If len(args)==1, the axis itself is returned. Otherwise, an
        :py:class:`AxisIndexer` which indexes over specified axes.

        """
        namemap = object.__getattribute__(self, '_namemap')
        axes = object.__getattribute__(self, '_axes')
        arr = object.__getattribute__(self, '_arr') 
        if len(args) == 1:
            return axes[namemap[args[0]]]
        else:
            return AxisIndexer(arr, *args)

class AxisIndexer(object):
    """
    An object which holds a reference to a DataArray and a list of axes and
    allows slicing by those axes.
    """
    # XXX don't support mapped indexing yet...
    def __init__(self, arr, *args):
        self.arr = arr
        self.axes = args
        axis_set = set(args)
        self._axis_map = [self.axes.index(axis.name) if axis.name in self.axes else None
            for axis in arr.axes]
    
    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = item,

        if len(item) != len(self.axes):
            raise ValueError("Incorrect slice length")
        
        slicer = tuple(
            item[self._axis_map[i]]
                if self._axis_map[i] is not None
                else slice(None, None, None)
            for i in range(len(self.arr.axes)))
    
        return self.arr[slicer]
        
class Axis(object):
    "Object to access a given axis of an array."
    # Key point: every axis contains a reference to its parent array!

    def __init__(self, name, index, parent_arr, labels=None):
        # Axis name should be a string or None
        if not isinstance(name, basestring) and name is not None:
            raise ValueError('name must be a string or None')
        self.name = name
        self.index = index
        self.parent_arr = parent_arr
        
        # If labels is not None, name should be defined
        if labels is not None and name is None:
            raise ValueError('labels only supported when Axis has a name')

        # This will raise if the labels are invalid:
        self._label_dict = self._validate_labels(labels)
        self.labels = labels

    def _copy(self, **kwargs):
        """
        Create a quick copy of this Axis without bothering to do
        label validation (these labels are already known as valid).

        Keyword args are replacements for constructor arguments

        Examples
        --------

        >>> a1 = Axis('time', 0, None, labels=[str(i) for i in range(10)])
        >>> a1
        Axis(name='time', index=0, labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        >>> a2 = a1._copy(labels=a1.labels[3:6])
        >>> a2
        Axis(name='time', index=0, labels=['3', '4', '5'])
        >>> a1 == a2
        False
        """
        name = kwargs.pop('name', self.name)
        index = kwargs.pop('index', self.index)
        parent_arr = kwargs.pop('parent_arr', self.parent_arr)
        cls = self.__class__ 
        ax = cls(name, index, parent_arr)

        labels = kwargs.pop('labels', copy.copy(self.labels))
        ax.labels = labels
        if labels is not None and len(labels) != len(self.labels):
            ax._label_dict = dict( zip(labels, range( len(labels) )) )
        else:
            ax._label_dict = copy.copy(self._label_dict)
        return ax

    # A guaranteed-to-be-a-string version of the axis name, which lets us
    # disambiguate when multiple unnamed axes exist in an array (since they all
    # have None for name).
    @property
    def _sname(self):
        if self.name is not None:
            return str(self.name)
        else:
            return "_%d" %  self.index

    def _validate_labels(self, labels):
        """Validate constraints on labels.

        Ensure:

        - uniqueness
        - length
        - no label is an integer
        """
        if labels is None:
            return None
        
        nlabels = len(labels)
        # XXX maybe Axis labels should be validated in __array_finalize__?

        # Sanity check: the first dimension must match that of the parent array
        if self.parent_arr is not None \
               and nlabels != self.parent_arr.shape[self.index]:
            e = 'Dimension mismatch between labels and data at index %i' % \
                self.index
            raise ValueError(e)

        # Validate types -- using generator for short circuiting
        if any( (isinstance(t, int) for t in labels) ):
            raise ValueError('Labels cannot be integers')
        
        # Validate uniqueness
        t_dict = dict(zip(labels, range(nlabels)))
        if len(t_dict) != nlabels:
            raise ValueError('non-unique label values not supported')
        return t_dict

    def set_name(self, name):
        # XXX: This makes some potentially scary changes to the parent
        #      array. It may end up being an insidious bug.

        # Axis name should be a string or None
        if not isinstance(name, basestring) and name is not None:
            raise ValueError('name must be a string or None')
        self.name = name
        pa = self.parent_arr
        nd = pa.ndim
        newaxes = [pa.axes[i] for i in range(self.index)]
        newaxes += [self]
        newaxes += [pa.axes[i] for i in range(self.index+1,nd)]
        _set_axes(pa, newaxes)
        
    def __len__(self):
        return self.parent_arr.shape[self.index]

    def __eq__(self, other):
        """
        Axes are equal iff they have matching names and indices. They
        do not need to have matching labels.

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
        """
        if not isinstance(other, self.__class__):
            return False

        return self.name == other.name and self.index == other.index and \
               self.labels == other.labels

    def __repr__(self):
        return 'Axis(name=%r, index=%i, labels=%r)' % \
               (self.name, self.index, self.labels)

    def __getitem__(self, key):
        """
        Return the item(s) of parent array along this axis as specified by `key`.

        `key` can be any of:
            - An integer
            - A tick
            - A slice of integers or ticks
            - `numpy.newaxis`, i.e. None

        Examples
        --------

        >>> A = DataArray(np.arange(2*3*2).reshape([2,3,2]), \
                ('a', ('b', ('b1','b2','b3')), 'c'))
        >>> b = A.axes.b

        >>> np.all(b['b1'] == A[:,0,:])
        True

        >>> np.all(b['b2':] == A[:,1:,:])
        True

        >>> np.all(b['b1':'b2'] == A[:,0:1,:])
        True
        """
        # XXX We don't handle fancy indexing at the moment
        if isinstance(key, (np.ndarray, list)):
            raise NotImplementedError('We do not handle fancy indexing yet')
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
            sli = self.make_slice(key)
            return np.ndarray.__getitem__(parent_arr, sli)
        
        # For other cases (slicing or scalar indexing of ndim>1 arrays),
        # build the proper slicing object to cut into the managed array
        fullslice = self.make_slice(key)
        # now get the translated key
        key = fullslice[self.index]
        out = np.ndarray.__getitem__(parent_arr, tuple(fullslice))

        newaxes = []
        for a in parent_arr.axes:
            newaxes.append( a._copy(parent_arr=parent_arr) )
        
        if isinstance(key, slice):
            # we need to find the labels, if any
            if self.labels:
                newlabels = self.labels[key]
            else:
                newlabels = None
            # insert new Axis with sliced labels
            newaxis = self._copy(parent_arr=parent_arr, labels=newlabels)
            newaxes[self.index] = newaxis

        if out.ndim < parent_arr_ndim:
            # We lost a dimension, drop the axis!
            newaxes = _pull_axis(newaxes, self)

        elif out.ndim > parent_arr_ndim:
            # We were indexed by a newaxis (None),
            # need to insert an unnamed axis before this axis.
            # Do this by inserting an Axis at the end of the axes, then
            # reindexing them
            new_axis = self.__class__(None, out.ndim-1, parent_arr)
            new_ax_order = [ax.index for ax in newaxes]
            new_ax_order.insert(self.index, out.ndim-1)
            newaxes.append(new_axis)
            newaxes = _reordered_axes(newaxes, new_ax_order)

        _set_axes(out, newaxes)
            
        return out

    def make_slice(self, key):
        """
        Make a slicing tuple into the parent array such that
        this Axis is cut up in the requested manner

        Parameters
        ----------
        key : a slice object, single label-like item, or None
          This slice object may have arbitrary types for .start, .stop,
          in which case label labels will be looked up. The .step attribute
          of course must be None or an integer.

        Returns
        -------
        keys : parent_arr.ndim-length tuple for slicing
        
        """

        full_slicing = [ slice(None) ] * self.parent_arr.ndim

        # if no labels, pop in the key and pray (will raise later)
        if not self.labels:
            full_slicing[self.index] = key
            return tuple(full_slicing)

        # in either case, try to translate slicing key
        if not isinstance(key, slice):
            lookups = (key,)
        else:
            lookups = (key.start, key.stop)
        
        looked_up = []
        for a in lookups:
            if a is None:
                looked_up.append(a)
                continue
            try:
                idx = self._label_dict[a]
            except KeyError:
                if not isinstance(a, int):
                    raise IndexError(
                        'Could not find an index to match %s'%str(a)
                        )
                idx = a
            looked_up.append(idx)

        # if not a slice object, then pop in the translated index and return
        if not isinstance(key, slice):
            full_slicing[self.index] = looked_up[0]
            return tuple(full_slicing)
        
        # otherwise, go for the step size now
        step = key.step
        if not isinstance(step, (int, type(None))):
            raise IndexError(
                'Slicing step size must be an integer or None, not %s'%str(step)
                )
        looked_up = looked_up + [step]
        new_key = slice(*looked_up)
        full_slicing[self.index] = new_key
        return tuple(full_slicing)
        
    def at(self, label):
        """
        Return data at a given label.

        >>> narr = DataArray(np.random.standard_normal((4,5)), axes=['a', ('b', 'abcde')])
        >>> arr = narr.axes.b['c']
        >>> arr.axes
        (Axis(name='a', index=0, labels=None),)
        """
        if not self.labels:
            raise ValueError('axis must have labels to extract data at a given label')
        slicing = self.make_slice(label)
        return self.parent_arr[slicing]
    
    def keep(self, labels):
        """
        Keep only certain labels of an axis.

        >>> narr = DataArray(np.random.standard_normal((4,5)),
        ...                  axes=['a', ('b', 'abcde')])
        >>> arr = narr.axes.b.keep('cd')
        >>> [a.labels for a in arr.axes]
        [None, 'cd']
        
        >>> arr.axes.a.at('label')
        Traceback (most recent call last):
            ...
        ValueError: axis must have labels to extract data at a given label
        """

        if not self.labels:
            raise ValueError('axis must have labels to keep certain labels')

        idxs = [self._label_dict[label] for label in labels]

        parent_arr = self.parent_arr # local for speed
        parent_arr_ndim = parent_arr.ndim

        fullslice = [slice(None)] * parent_arr_ndim
        fullslice[self.index] = idxs
        out = np.ndarray.__getitem__(parent_arr, tuple(fullslice))

        # just change the current axes
        new_axes = [a._copy() for a in out.axes]
        new_axes[self.index] = self._copy(labels=labels)
        _set_axes(out, new_axes)
        return out

    def drop(self, labels):
        """
        Keep only certain labels of an axis.

        Example
        =======
        >>> darr = DataArray(np.random.standard_normal((4,5)),
        ...                  axes=['a', ('b', ['a','b','c','d','e'])])
        >>> arr1 = darr.axes.b.keep(['c','d'])
        >>> arr2 = darr.axes.b.drop(['a','b','e'])
        >>> np.all(arr1 == arr2)
        True
        """

        if not self.labels:
            raise ValueError('axis must have labels to drop labels')

        kept = [t for t in self.labels if t not in labels]
        return self.keep(kept)

    def __int__(self):
        return self.index
# -- Axis utilities ------------------------------------------------------------

def _names_to_numbers(axes, ax_ids):
    """
    Convert any axis names to axis indices. Pass through any integer ax_id,
    and convert to integer any ax_id that is an Axis.
    """
    proc_ids = []
    for ax_id in ax_ids:
        if isinstance(ax_id, basestring):
            matches = [ax for ax in axes if ax._sname == ax_id]
            if not matches:
                raise NamedAxisError('No axis named %s' % ax_id)
            proc_ids.append(matches[0].index)
        else:
            proc_ids.append(int(ax_id))
    return proc_ids

def _validate_axes(arr):
    # This should always be true our axis lists....
    assert all(i == a.index and arr is a.parent_arr 
            for i,a in enumerate(arr.axes))

def _pull_axis(axes, target_axis):
    """
    Return axes removing any axis matching `target_axis`. A match
    is determined by the Axis.index
    """
    newaxes = []
    if isinstance(target_axis, (list, tuple)):
        pulled_indices = [ax.index for ax in target_axis]
    else:
        pulled_indices = [target_axis.index]
    c = 0
    for a in axes:
        if a.index not in pulled_indices:
            newaxes.append(a._copy(index=c))
            c += 1
    return newaxes    

def _set_axes(dest, in_axes):
    """
    Set the axes in `dest` from `in_axes`.

    WARNING: The destination is modified in-place! The following attribute
    is added to it:

    - axes: an instance of AxesManager which manages access to axes.

    Parameters
    ----------
      dest : array
      in_axes : sequence of axis objects
    """
    # XXX: This method is called multiple times during a DataArray's lifetime.
    #      Should rethink exactly when Axis copies need to be made
    axes = []
    ax_holder = KeyStruct()
    # Create the containers for various axis-related info
    for ax in in_axes:
        new_ax = ax._copy(parent_arr=dest)
        axes.append(new_ax)
        if hasattr(ax_holder, ax._sname):
            raise NamedAxisError( """There is another Axis in this group with
                    the same name""")
        ax_holder[ax._sname] = new_ax
    # Store these containers as attributes of the destination array
    dest.axes = AxesManager(dest, axes)

def names2namedict(names):
    """Make a name map out of any name input.
    """
    raise NotImplementedError() 

# -- Method Wrapping -----------------------------------------------------------

# XXX: Need to convert from positional arguments to named arguments

def _apply_reduction(opname, kwnames):
    """
    Wraps the reduction operator with name `opname`. Must supply the
    method keyword argument names, since in many cases these methods
    are called with the keyword args as positional args
    """
    super_op = getattr(np.ndarray, opname)
    if 'axis' not in kwnames:
        raise ValueError(
            'The "axis" keyword must be part of an ndarray reduction signature'
            )
    def runs_op(*args, **kwargs):
        inst = args[0]
        # re/place any additional args in the appropriate keyword arg
        for nm, val in zip(kwnames, args[1:]):
            kwargs[nm] = val
        axis = kwargs.pop('axis', None)

        if not isinstance(inst, DataArray) or axis is None:
            # do nothing special if not a DataArray, otherwise
            # this is a full reduction, so we lose all axes
            return super_op(np.asarray(inst), **kwargs)

        axes = list(inst.axes)
        # try to convert a named Axis to an integer..
        # don't try to catch an error
        axis_idx = _names_to_numbers(inst.axes, [axis])[0]
        if not kwargs.get('keepdims', False):
            axes = _pull_axis(axes, inst.axes[axis_idx])
        kwargs['axis'] = axis_idx
        arr = super_op(inst, **kwargs)
        if not is_numpy_scalar(arr):
            _set_axes(arr, axes)
        return arr
    runs_op.__name__ = opname
    runs_op.__doc__ = super_op.__doc__
    return runs_op

def is_numpy_scalar(arr):
    return arr.ndim == 0

def _apply_accumulation(opname, kwnames):
    super_op = getattr(np.ndarray, opname)
    if 'axis' not in kwnames:
        raise ValueError(
            'The "axis" keyword must be part of an ndarray reduction signature'
            )
    def runs_op(*args, **kwargs):
        inst = args[0]
        
        # re/place any additional args in the appropriate keyword arg
        for nm, val in zip(kwnames, args[1:]):
            kwargs[nm] = val
        axis = kwargs.pop('axis', None)
        if axis is None:
            # this will flatten the array and lose all dimensions
            return super_op(np.asarray(inst), **kwargs)

        # try to convert a named Axis to an integer..
        # don't try to catch an error
        axis_idx = _names_to_numbers(inst.axes, [axis])[0]
        kwargs['axis'] = axis_idx
        return super_op(inst, **kwargs)
    runs_op.__name__ = opname
    runs_op.__doc__ = super_op.__doc__
    return runs_op
            
class DataArray(np.ndarray):
    # XXX- we need to figure out where in the numpy C code .T is defined!
    @property
    def T(self):
        return self.transpose()

    def __new__(cls, data, axes=None, dtype=None, copy=False):
        # XXX if an entry of axes is a tuple, it is interpreted
        # as a (name, labels) tuple 
        # Ensure the output is an array of the proper type
        arr = np.array(data, dtype=dtype, copy=copy).view(cls)

        if axes is None:
            if hasattr(data,'axes'):
                _set_axes(arr, data.axes)
                return arr
            axes = []

        elif len(axes) > arr.ndim:
            raise NamedAxisError('Axes list should have length <= array ndim')
        
        # Pad axes spec to match array shape
        axes = list(axes) + [None]*(arr.ndim - len(axes))

        axlist = []
        for i, axis_spec in enumerate(axes):
            if isinstance(axis_spec, basestring) or axis_spec is None:
                # string name
                name = axis_spec
                labels = None
            else:
                if len(axis_spec) != 2:
                    raise ValueError("""If the axis specification is a tuple,
                            it must be of the form (name, labels)""")
                name, labels = axis_spec
            axlist.append(Axis(name, i, arr, labels=labels))

        _set_axes(arr, axlist)
        _validate_axes(arr)

        return arr

    def set_name(self, i, name):
        self.axes[i].set_name(name)

    @property
    def names (self):
        """Returns a tuple with all the axis names."""
        return tuple((ax.name for ax in self.axes))
    
    def index_by(self, *args):
        return AxisIndexer(self, *args)

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
        
##         print "finalizing DataArray" # dbg
        
        # Ref: see http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        
        # provide info for what's happening
##         print "finalize:\t%s\n\t\t%s" % (self.__class__, obj.__class__) # dbg
##         print "obj     :", obj.shape  # dbg
        # provide more info
        if obj is None: # own constructor, we're done
            return
        if not hasattr(obj, 'axes'): # looks like view cast
            _set_axes(self, [])
            return
        # new-from-template: we just copy the axes from the template,
        # and hope the calling rountine knows what to do with the output
##         print 'setting axes on self from obj' # dbg
        _set_axes(self, obj.axes)
            
        # validate the axes
        _validate_axes(self)

    def __array_prepare__(self, obj, context=None):
        "Called at the beginning of each ufunc."

##         print "preparing DataArray" # dbg

        # Ref: see http://docs.scipy.org/doc/numpy/reference/arrays.classes.html

        # provide info for what's happening
        #print "prepare:\t%s\n\t\t%s" % (self.__class__, obj.__class__) # dbg
        #print "obj     :", obj.shape  # dbg
        #print "context :", context  # dbg
        
        if context is not None and len(context[1]) > 1:
            "binary ufunc operation"
            other = context[1][1]
##             print "other   :", other.__class__

            if not isinstance(other,DataArray):
                return obj
            
##                 print "found DataArray, comparing axes"

            # walk back from the last axis on each array, check
            # that the name and shape are acceptible for broadcasting
            these_axes = list(self.axes)
            those_axes = list(other.axes)
            #print self.shape, self.names # dbg
            while these_axes and those_axes:
                that_ax = those_axes.pop(-1)
                this_ax = these_axes.pop(-1)
                # print self.shape # dbg
                this_dim = self.shape[this_ax.index]
                that_dim = other.shape[that_ax.index]
                if that_ax.name != this_ax.name:
                    # A valid name can be mis-matched IFF the other
                    # (name, length) pair is:
                    # * (None, 1)
                    # * (None, {this,that}_dim).                    
                    # In this case, the unnamed Axis should
                    # adopt the name of the matching Axis in the
                    # other array (handled in elsewhere)
                    if that_ax.name is not None and this_ax.name is not None:
                        raise NamedAxisError(
                            'Axis axes are incompatible for '\
                            'a binary operation: ' \
                            '%s, %s'%(self.names, other.names))
                if that_ax.labels != this_ax.labels:
                    if that_ax.labels is not None and this_ax.labels is not None:
                        raise NamedAxisError(
                            'Axis labels are incompatible for '\
                            'a binary operation.')

                # XXX: Does this dimension compatibility check happen
                #      before __array_prepare__ is even called? This
                #      error is not fired when there's a shape mismatch.
                if this_dim==1 or that_dim==1 or this_dim==that_dim:
                    continue
                raise NamedAxisError('Dimension with name %s has a '\
                                     'mis-matched shape: ' \
                                     '(%d, %d) '%(this_ax.name,
                                                  this_dim,
                                                  that_dim))
        return obj
                    

    def __array_wrap__(self, obj, context=None):
        # provide info for what's happening
        # print "prepare:\t%s\n\t\t%s" % (self.__class__, obj.__class__) # dbg
        # print "obj     :", obj.shape  # dbg
        # print "context :", context # dbg

        other = None
        if context is not None and len(context[1]) > 1:
            "binary ufunc operation"
            other = context[1][1]
##             print "other   :", other.__class__
            
        if isinstance(other,DataArray):            
##                 print "found DataArray, comparing names"

            # walk back from the last axis on each array to get the
            # correct names/labels
            these_axes = list(self.axes)
            those_axes = list(other.axes)
            ax_spec = []
            while these_axes and those_axes:
                this_ax = these_axes.pop(-1)
                that_ax = those_axes.pop(-1)
                # If we've broadcasted this array against another, then
                # this_ax.name may be None, in which case the new array's
                # Axis name should take on the value of that_ax
                if this_ax.name is None:
                    ax_spec.append(that_ax)
                else:
                    ax_spec.append(this_ax)
            ax_spec = ax_spec[::-1]
            # if the axes are not totally consumed on one array or the other,
            # then grab those names/labels for the rest of the dims
            if these_axes:
                ax_spec = these_axes + ax_spec
            elif those_axes:
                ax_spec = those_axes + ax_spec
        else:
            ax_spec = self.axes

        res = obj.view(type(self))
        new_axes = []
        for i, ax in enumerate(ax_spec):
            new_axes.append( ax._copy(index=i, parent_arr=res) )
        _set_axes(res, new_axes)
        return res
                
    def __getitem__(self, key):
        """Support x[k] access."""
        # Slicing keys:
        # * a single int
        # * a single newaxis
        # * a tuple with length <= self.ndim (may have newaxes)
        # * a tuple with length > self.ndim (MUST have newaxes)
        # * list, array, etc for fancy indexing (not implemented)
        
        # Cases
        if isinstance(key, list) or isinstance(key, np.ndarray):
            # fancy indexing
            # XXX need to be cast to an "ordinary" ndarray
            raise NotImplementedError
        if key is None:
            key = (key,)

        if isinstance(key, tuple):
            old_shape = self.shape
            old_axes = self.axes
            new_shape, new_axes, key = _make_singleton_axes(self, key)
            # Will undo this later
            self.shape = new_shape
            _set_axes(self, new_axes)

            # Pop the axes off in descending order to prevent index renumbering
            # headaches.
            reductions = list(reversed(sorted(zip(key, new_axes),
                                              key=lambda t: t[1].index)))
            arr = self
            for k,ax in reductions:
                arr = arr.axes[ax.index][k]

            # restore old shape and axes
            self.shape = old_shape
            _set_axes(self, old_axes)
        else:
            arr = self.axes[0][key]

        return arr

    def __str_repr_helper(self, ary_repr):
        """Helper function for __str__ and __repr__. Produce a text
        representation of the axis suitable for eval() as an argument to a
        DataArray constructor."""
        axis_spec = repr(tuple(ax.name if ax.labels is None 
            else (ax.name, tuple(ax.labels)) for ax in self.axes))
        return "%s(%s,\n%s)" % \
                (self.__class__.__name__, ary_repr, axis_spec)

    def __str__(self):
        return self.__str_repr_helper(np.asarray(self).__str__())

    def __repr__(self):
        return self.__str_repr_helper(np.asarray(self).__repr__())

    # Methods from ndarray

    def transpose(self, *axes):
        # implement tuple-or-*args logic of np.transpose
        axes = list(axes)
        if not axes:
            axes = list(range(self.ndim-1,-1,-1))
        # expand sequence if sequence passed as first and only arg
        elif len(axes) < self.ndim:
            try:
                axes = list(axes[0])
            except TypeError:
                pass
        proc_axids = _names_to_numbers(self.axes, axes)
        out = np.ndarray.transpose(self, proc_axids)
        _set_axes(out, _reordered_axes(self.axes, proc_axids, parent=out))
        return out
    transpose.__doc__ = np.ndarray.transpose.__doc__

    def swapaxes(self, axis1, axis2):
        # form a transpose operation with axes specified
        # by (axis1, axis2) swapped
        axis1, axis2 = _names_to_numbers(self.axes, [axis1, axis2])
        ax_idx = list(range(self.ndim))
        tmp = ax_idx[axis1]
        ax_idx[axis1] = ax_idx[axis2]
        ax_idx[axis2] = tmp
        out = np.ndarray.transpose(self, ax_idx)
        _set_axes(out, _reordered_axes(self.axes, ax_idx, parent=out))
        return out
    swapaxes.__doc__ = np.ndarray.swapaxes.__doc__

    def ptp(self, axis=None, out=None):
        mn = self.min(axis=axis)
        mx = self.max(axis=axis, out=out)
        if isinstance(mn, np.ndarray):
            mx -= mn
            return mx
        else:
            return mx-mn
    ptp.__doc__ = np.ndarray.ptp.__doc__

    # -- Various extraction and reshaping methods ----------------------------
    def diagonal(self, *args, **kwargs):
        # reverts to being an ndarray
        args = (np.asarray(self),) + args
        return np.diagonal(*args, **kwargs)
    diagonal.__doc__ = np.ndarray.diagonal.__doc__
    
    def flatten(self, **kwargs):
        # reverts to being an ndarray
        return np.asarray(self).flatten(**kwargs)
    flatten.__doc__ = np.ndarray.flatten.__doc__

    def ravel(self, **kwargs):
        # reverts to being an ndarray
        return np.asarray(self).ravel(**kwargs)
    ravel.__doc__ = np.ndarray.ravel.__doc__

    def repeat(self, *args, **kwargs):
        raise NotImplementedError

    def squeeze(self):
        axes = list(self.axes)
        pinched_axes = [x for x in axes if self.shape[x.index] == 1]
        squeezed_shape = [d for d in self.shape if d > 1]
        axes = _pull_axis(axes, pinched_axes)
        arr = self.reshape(squeezed_shape)
        _set_axes(arr, axes)
        return arr

    def reshape(self, *args, **kwargs):
        # XXX:
        # * reshapes such as a.reshape(a.shape + (1,)) will be supported
        # * reshapes such as a.ravel() will return ndarray
        # * reshapes such as a.reshape(x', y', z') ???
        # print 'reshape called', args, kwargs # dbg
        if len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                args = args[0]
            else:
                return np.asarray(self).reshape(*args)
        # if adding/removing length-1 dimensions, then add an unnamed Axis
        # or pop an Axis
        old_shape = list(self.shape)
        new_shape = list(args)
        old_non_single_dims = [d for d in old_shape if d > 1]
        new_non_single_dims = [d for d in new_shape if d > 1]
        axes_to_pull = []
        axes = list(self.axes)
        if old_non_single_dims == new_non_single_dims:
            # pull axes first
            i = j = 0
            while i < len(new_shape) and j < len(old_shape):
                if new_shape[i] != old_shape[j] and old_shape[j] == 1:
                    axes_to_pull.append(self.axes[j])
                else:
                    i += 1
                j += 1
            # pull anything that extends past the length of the new shape
            axes_to_pull += [self.axes[i] for i in range(j, len(old_shape))]
            old_shape = [self.shape[ax.index]
                         for ax in axes if ax not in axes_to_pull]
            axes = _pull_axis(axes, axes_to_pull)
            # now append axes
            i = j = 0
            axes_order = []
            while i < len(new_shape) and j < len(old_shape):
                if new_shape[i] != old_shape[j] and new_shape[i] == 1:
                    idx = len(axes)
                    axes.append( Axis(None, idx, self) )
                    axes_order.append(idx)
                else:
                    axes_order.append(j)
                    j += 1
                i += 1
            # append None axes for all shapes past the length of the old shape
            new_idx = range(i, len(new_shape))
            axes += [Axis(None, idx, self) for idx in new_idx]
            axes_order += new_idx
            axes = _reordered_axes(axes, axes_order)
            arr = super(DataArray, self).reshape(*new_shape)
            _set_axes(arr, axes)
            return arr

        # if dimension sizes can be moved around between existing axes,
        # then go ahead and try to keep the Axis meta-data
        raise NotImplementedError
    
    # -- Sorting Ops ---------------------------------------------------------
    # ndarray sort with axis==None flattens the array: return ndarray
    
    # Otherwise, if there are labels at the axis in question, then
    # the sample-to-label correspondence becomes inconsistent across
    # the remaining axes. Also return a plain ndarray.
    
    # Otherwise, order the axis in question--default axis is -1

    # XXX: Might be best to always return ndarray, since the return
    #      type is so inconsistent
    def sort(self, **kwargs):
        axis = kwargs.get('axis', -1)
        if axis is not None:
            axis = _names_to_numbers(self.axes, [axis])[0]
            kwargs['axis'] = axis
        if axis is None or self.axes[axis].labels:
            # Returning NEW ndarray
            arr = np.asarray(self).copy()
            arr.sort(**kwargs)
            return arr
        # otherwise, just do the op on this array
        super(DataArray, self).sort(**kwargs)

    def argsort(self, **kwargs):
        axis = kwargs.get('axis', -1)
        if axis is not None:
            axis = _names_to_numbers(self.axes, [axis])[0]
            kwargs['axis'] = axis
        if axis is None or self.axes[axis].labels:
            # Returning NEW ndarray
            arr = np.asarray(self)
            return arr.argsort(**kwargs)
        # otherwise, just do the op on this array
        axes = list(self.axes)
        arr = super(DataArray, self).argsort(**kwargs)
        _set_axes(arr, axes)
        return arr

    # -- Reductions ----------------------------------------------------------
    mean = _apply_reduction('mean', ('axis', 'dtype', 'out', 'keepdims'))
    var = _apply_reduction('var', ('axis', 'dtype', 'out', 'ddof', 'keepdims'))

    def std(self, *args, **kwargs):
        ret = self.var(*args, **kwargs)
        if isinstance(ret, np.ndarray):
            ret = np.sqrt(ret, out=ret)
        elif hasattr(ret, 'dtype'):
            ret = ret.dtype.type(np.sqrt(ret))
        else:
            ret = np.sqrt(ret)
        return ret
    std.__doc__ = np.ndarray.std.__doc__

    min = _apply_reduction('min', ('axis', 'out', 'keepdims'))
    max = _apply_reduction('max', ('axis', 'out', 'keepdims'))

    sum = _apply_reduction('sum', ('axis', 'dtype', 'out', 'keepdims'))
    prod = _apply_reduction('prod', ('axis', 'dtype', 'out', 'keepdims'))

    all = _apply_reduction('all', ('axis', 'dtype', 'out', 'keepdims'))
    any = _apply_reduction('any', ('axis', 'dtype', 'out', 'keepdims'))

    ### these change the meaning of the axes..
    ### should probably return ndarrays
    argmax = _apply_reduction('argmax', ('axis', 'out'))
    argmin = _apply_reduction('argmin', ('axis', 'out'))

    # -- Accumulations -------------------------------------------------------
    cumsum = _apply_accumulation('cumsum', ('axis', 'dtype', 'out'))
    cumprod = _apply_accumulation('cumprod', ('axis', 'dtype', 'out'))

# -- DataArray utilities -------------------------------------------------------

def _reordered_axes(axes, axis_indices, parent=None):
    ''' Perform axis reordering according to `axis_indices`
    Checks to ensure that all axes have the same parent array.
    Parameters
    ----------
    axes : sequence of axes
       The axis indices in this list reflect the axis ordering before
       the permutation given by `axis_indices`
    axis_indices : sequence of ints
       indices giving new order of axis numbers
    parent : ndarray or None
       if not None, used as parent for all created axes

    Returns
    -------
    ro_axes : sequence of axes
       sequence of axes (with the same parent array)
       in arbitrary order with axis indices reflecting
       reordering given by `axis_indices`

    Examples
    --------
    >>> a = Axis('x', 0, None)
    >>> b = Axis('y', 1, None)
    >>> c = Axis(None, 2, None)
    >>> res = _reordered_axes([a,b,c], (1,2,0))
    '''

    new_axes = []
    for new_ind, old_ind in enumerate(axis_indices):
        ax = axes[old_ind]
        if parent is None:
            parent_arr = ax.parent_arr
        else:
            parent_arr = parent
        new_ax = ax._copy(index=new_ind, parent_arr=parent_arr)
        new_axes.append(new_ax)
    return new_axes

def _expand_ellipsis(key, ndim):
    "Expand the slicing tuple if the Ellipsis object is present."
    # Ellipsis can only occur once (not totally the same as NumPy),
    # which apparently allows multiple Ellipses to follow one another
    kl = list(key)
    ecount = kl.count(Ellipsis)
    if ecount > 1:
        raise IndexError('invalid index')
    if ecount < 1:
        return key
    e_index = kl.index(Ellipsis)
    kl_end = kl[e_index+1:] if e_index < len(key)-1 else []
    kl_beg = kl[:e_index]
    kl_middle = [slice(None)] * (ndim - len(kl_end) - len(kl_beg))
    return tuple( kl_beg + kl_middle + kl_end )

def _make_singleton_axes(arr, key):
    """
    Parse the slicing key to determine whether the array should be
    padded with singleton dimensions prior to slicing. Also expands
    any Ellipses in the slicing key.

    Parameters
    ----------
    arr : DataArray
    key : slicing tuple

    Returns
    -------
    (shape, axes, key)

    These are the new shape, with singleton axes included; the new axes,
    with an unnamed Axis at each singleton dimension; and the new
    slicing key, with `newaxis` keys replaced by slice(None)
    """
    
    key = _expand_ellipsis(key, arr.ndim)
    if len(key) <= arr.ndim and None not in key:
        return arr.shape, arr.axes, key

    # The full slicer will be length=arr.ndim + # of dummy-dims..
    # Boost up the slices to full "rank" ( can cut it down later for savings )
    n_new_dims = len([x for x in key if x is None])
    key = key + (slice(None),) * (arr.ndim + n_new_dims - len(key))
    # wherever there is a None in the key,
    # * replace it with slice(None)
    # * place a new dimension with length 1 in the shape,
    # * and add a new unnamed Axis to the axes
    new_dims = []
    new_key = []
    d_cnt = 0
    new_ax_pos = arr.ndim
    new_axes = list(arr.axes)
    ax_order = []
    for k in key:
        if k is None:
            new_key.append(slice(None))
            new_dims.append(1)
            # add a new Axis at the end of the list, then reorder
            # the list later to ensure the Axis indices are accurate
            new_axes.append(Axis(None, new_ax_pos, arr))
            ax_order.append(new_ax_pos)
            new_ax_pos += 1
        else:
            new_key.append(k)
            try:
                new_dims.append(arr.shape[d_cnt])
                ax_order.append(d_cnt)
                d_cnt += 1
            except IndexError:
                raise IndexError('too many indices')
    ro_axes = _reordered_axes(new_axes, ax_order)
    # Cut down all trailing "slice(None)" objects at the end of the new key.
    # (But! it seems we have to leave in at least one slicing element
    #  in order to get a new array)
    while len(new_key)>1 and new_key[-1] == slice(None):
        new_key.pop()
    return tuple(new_dims), ro_axes, tuple(new_key)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

