#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import copy

import numpy as np
import nose.tools as nt

from stuple import *

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
    def __init__(self, label, index, parent_arr, ticks=None):
        # Axis should be a string or None
        if not isinstance(label, basestring) and label is not None:
            raise ValueError('label must be a string or None')
        self.label = label
        self.index = index
        self.parent_arr = parent_arr

        # If ticks is not None, label should be defined
        if ticks is not None and label is None:
            raise ValueError('ticks only supported when Axis has a label')

        # This will raise if the ticks are invalid:
        self._tick_dict, self._tick_dict_reverse = self._validate_ticks(ticks)
        self.ticks = ticks

    @property
    def name(self):
        if self.label is not None:
            return str(self.label)
        else:
            return "_%d" %  self.index

    def _validate_ticks(self, ticks, check_length=False):
        """Validate constraints on ticks.

        Ensure:

        - uniqueness
        - length
        """
        if ticks is None:
            return None, None
        # We always store ticks as numpy arrays
        #ticks = np.asarray(ticks)

        nticks = len(ticks)
        # Sanity check: the first dimension must match that of the parent array
        # XXX this causes exceptions when slicing:
        # XXX maybe ticks of each axis should be validated in __array_finalize__?

        if check_length and nticks != self.parent_arr.shape[self.index]:
            e = "Dimension mismatch between ticks and data at index %i" % \
                self.index
            raise ValueError(e)
        
        # Validate uniqueness
        t_dict = dict(zip(ticks, xrange(nticks)))
        if len(t_dict) != nticks:
            raise ValueError("non-unique tick values not supported")
        t_dict_reverse = dict(zip(xrange(nticks), ticks))
        return t_dict, t_dict_reverse
        
    def __len__(self):
        return self.parent_arr.shape[self.index]

    def __eq__(self, other):
        ''' Axes are equal iff they have matching labels and indices

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
        return self.label == other.label and self.index == other.index

    def __str__(self):
        return 'Axis(label=%r, index=%i, ticks=%r)' % \
               (self.label, self.index, self.ticks)

    __repr__ = __str__
    
    def __getitem__(self, key):
        # `key` can be one of:
        # * integer (more generally, any valid scalar index)
        # * slice
        # * np.newaxis (ie, None)
        # * list (fancy indexing)
        # * array (fancy indexing)
        #
        # XXX We don't handle fancy indexing at the moment
        if isinstance(key, (np.ndarray, list)):
            raise NotImplementedError('We do not handle fancy indexing yet')
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
            return np.ndarray.__getitem__(parent_arr, key)

        # XXX TODO - fancy indexing
        # For other cases (slicing or scalar indexing of ndim>1 arrays),
        # build the proper slicing object to cut into the managed array
        fullslice = [slice(None)] * parent_arr_ndim
        fullslice[self.index] = key
        #print 'getting output'  # dbg
        out = np.ndarray.__getitem__(parent_arr, tuple(fullslice))
        #print 'returning output'  # dbg

        if isinstance(key, slice):
            newaxes = []
            # we need to find the ticks, if any
            if self.ticks:
                newticks = self.ticks[key]
            else:
                newticks = None
            newaxis = self.__class__(self.label, self.index, parent_arr, ticks=newticks)
            newaxes = []
            for a in parent_arr.axes:
                newaxes.append(a.__class__(a.label, a.index, parent_arr, ticks=a.ticks))
            newaxes[self.index] = newaxis

            _set_axes(out, newaxes)

        if out.ndim < parent_arr_ndim:
            # We lost a dimension, drop the axis!
##             print 'Dropping axes' # dbg
            _set_axes(out, _pull_axis(parent_arr.axes, self))
        elif out.ndim > parent_arr_ndim:
            # We were indexed by a newaxis (None),
            # need to insert an unlabeled axis before this axis
            new_axis = self.__class__(None, out.ndim-1, parent_arr)
            newaxes = parent_arr.axes[:]
            new_ax_order = [ax.index for ax in newaxes]
            new_ax_order.insert(self.index, out.ndim-1)
            newaxes.append(new_axis)
            ro_axes = _reordered_axes(newaxes, new_ax_order)
            _set_axes(out, ro_axes)
            
        return out
        
    def at(self, tick):
        """
        Return data at a given tick.

        >>> narr = DataArray(np.random.standard_normal((4,5)), labels=['a', ('b', 'abcde')])
        >>> arr = narr.axis.b.at('c')
        >>> arr.axes
        [Axis(label='a', index=0, ticks=None)]
        >>>     

        """
        if not self.ticks:
            raise ValueError('axis must have ticks to extract data at a given tick')

        # the index of the tick in the axis
        try:
            idx = self._tick_dict[tick]
        except KeyError:
            raise KeyError('tick %s not found in axis "%s"' % (`tick`, self.name))

        parent_arr = self.parent_arr # local for speed
        parent_arr_ndim = parent_arr.ndim

        fullslice = [slice(None)] * parent_arr_ndim
        fullslice[self.index] = idx
        out = np.ndarray.__getitem__(parent_arr, tuple(fullslice))

        # we will have lost a dimension and drop the current axis
        _set_axes(out, _pull_axis(parent_arr.axes, self))
        return out

    def keep(self, ticks):
        """
        Keep only certain ticks of an axis.

        >>> narr = DataArray(np.random.standard_normal((4,5)),
        ...                  labels=['a', ('b', 'abcde')])
        >>> arr = narr.axis.b.keep('cd')
        >>> [a.ticks for a in arr.axes]
        [None, 'cd']
        
        >>> arr.axis.a.at('tick')
        Traceback (most recent call last):
        ...
        ValueError: axis must have ticks to extract data at a given tick
        """

        if not self.ticks:
            raise ValueError('axis must have ticks to keep certain ticks')

        idxs = [self._tick_dict[tick] for tick in ticks]

        parent_arr = self.parent_arr # local for speed
        parent_arr_ndim = parent_arr.ndim

        fullslice = [slice(None)] * parent_arr_ndim
        fullslice[self.index] = idxs
        out = np.ndarray.__getitem__(parent_arr, tuple(fullslice))

        # just change the current axes

        new_axes = [Axis(a.label, a.index, a.parent_arr, ticks=a.ticks) for a in out.axes]
        new_axes[self.index] = Axis(self.label, self.index, self.parent_arr, ticks=ticks)
        _set_axes(out, new_axes)
        return out

    def drop(self, ticks):
        """
        Keep only certain ticks of an axis.

        >>> narr = DataArray(np.random.standard_normal((4,5)),
        ...                  labels=['a', ('b', 'abcde')])
        >>> arr1 = narr.axis.b.keep('cd')
        >>> arr2 = narr.axis.b.drop('abe')
        >>> np.alltrue(np.equal(arr1, arr2))
        True
        """

        if not self.ticks:
            raise ValueError('axis must have ticks to drop ticks')

        kept = [t for t in self.ticks if t not in ticks]
        return self.keep(kept)

    def __int__(self):
        return self.index

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



def _validate_axes(axes):
    """
    This should always be true our axis lists....
    """
    p = axes[0].parent_arr
    for i, a in enumerate(axes):
        nt.assert_equals(i, a.index)
        nt.assert_true(p is a.parent_arr)
        ## if a.ticks:
        ##     a._validate_ticks(a.ticks, check_length=True)

def _pull_axis(axes, target_axis):
    ''' Return axes removing any axis matching `target_axis`'''
    newaxes = []
    parent_arr = target_axis.parent_arr
    c = 0
    for a in axes:
        if a.index != target_axis.index:
            newaxes.append(a.__class__(a.label, c, parent_arr, ticks=a.ticks))
            c += 1
    return newaxes    

def _set_axes(dest, in_axes):
    """Set the axes in `dest` from `in_axes`.

    WARNING: The destination is modified in-place!  The following attributes
    are added to it:

    - axis: a KeyStruct with each axis as a named attribute.
    - axes: a list of all axis instances.
    - labels: a list of all the axis labels.

    Parameters
    ----------
      dest : array
      in_axes : sequence of axis objects
    """
    # XXX here is where the logic is implemented for missing names.
    # Here there are no named axis objects if there are fewer names than
    # axes in the array
    axes = []
    labels = []
    ax_holder = KeyStruct()
    # Create the containers for various axis-related info
    for ax in in_axes:
        new_ax = ax.__class__(ax.label, ax.index, dest, ticks=ax.ticks)
        axes.append(new_ax)
        labels.append(ax.label)
        ax_holder[ax.name] = new_ax
    # Store these containers as attributes of the destination array
    dest.axes = axes
    dest.labels = labels
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

    def __new__(cls, data, labels=None, dtype=None, copy=False):
        # XXX if an entry of labels is a tuple, it is interpreted
        # as a (label, ticks) tuple 
        # Ensure the output is an array of the proper type
        arr = np.array(data, dtype=dtype, copy=copy).view(cls)
        if labels is None:
            if hasattr(data,'axes'):
                _set_axes(arr, data.axes)
                return arr
            labels = []
        elif len(labels) > arr.ndim:
            raise NamedAxisError("labels list should have length <= array ndim")
        
        labels = list(labels) + [None]*(arr.ndim - len(labels))
        axes = []
        for i, label_spec in enumerate(labels):
            if type(label_spec) == type(()):
                if len(label_spec) != 2:
                    raise ValueError("if the label specification is a tuple, it must be of the form (label, ticks)")
                label, ticks = label_spec
            else:
                label = label_spec
                ticks = None
            axes.append(Axis(label, i, arr, ticks=ticks))

        _set_axes(arr, axes)

        # validate the axes
        _validate_axes(axes)

        return arr

    @property
    def aix(self):
        # Returns an anonymous slicing tuple that knows
        # about this array's geometry
        return stuple( ( slice(None), ) * self.ndim,
                       axes = self.axes )

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
        # new-from-template: we just copy the labels from the template,
        # and hope the calling rountine knows what to do with the output
##         print 'setting axes on self from obj' # dbg
        _set_axes(self, obj.axes)
            
        # validate the axes
        _validate_axes(self.axes)

    def __array_prepare__(self, obj, context=None):
        """Called at the beginning of each ufunc.
        """

##         print "preparing DataArray" # dbg

        # Ref: see http://docs.scipy.org/doc/numpy/reference/arrays.classes.html

        # provide info for what's happening
        print "prepare:\t%s\n\t\t%s" % (self.__class__, obj.__class__) # dbg
        print "obj     :", obj.shape  # dbg
        print "context :", context
        
        if context is not None and len(context[1]) > 1:
            "binary ufunc operation"
            other = context[1][1]
##             print "other   :", other.__class__

            if not isinstance(other,DataArray):
                return obj
            
##                 print "found DataArray, comparing labels"

            # walk back from the last axis on each array, check
            # that the label and shape are acceptible for broadcasting
            these_axes = self.axes[:]
            those_axes = other.axes[:]
            while these_axes and those_axes:
                that_ax = those_axes.pop(-1)
                this_ax = these_axes.pop(-1)
                this_dim = self.shape[this_ax.index]
                that_dim = other.shape[that_ax.index]
                if that_ax.label != this_ax.label:
                    # A valid label can be mis-matched IFF the other
                    # (label, length) pair is (None, 1).
                    # In this case, the singleton dimension should
                    # adopt the label of the matching dimension in the
                    # other array
                    if (that_ax.label, that_dim) != (None, 1) and \
                       (this_ax.label, this_dim) != (None, 1):
                        raise NamedAxisError(
                            'Axis labels are incompatible for '\
                            'a binary operation: ' \
                            '%s, %s'%(self.labels, other.labels)
                            )

                # XYZ: Does this dimension compatibility check happen
                # before __array_prepare__ is even called???
                if this_dim==1 or that_dim==1 or this_dim==that_dim:
                    continue
                raise NamedAxisError('Dimension with label %s has a '\
                                     'mis-matched shape: ' \
                                     '(%d, %d) '%(this_ax.label,
                                                  this_dim,
                                                  that_dim))
        return obj
                    

    def __array_wrap__(self, obj, context=None):
        # provide info for what's happening
        print "prepare:\t%s\n\t\t%s" % (self.__class__, obj.__class__) # dbg
        print "obj     :", obj.shape  # dbg
        print "context :", context

        other = None
        if context is not None and len(context[1]) > 1:
            "binary ufunc operation"
            other = context[1][1]
##             print "other   :", other.__class__
            
        if isinstance(other,DataArray):            
##                 print "found DataArray, comparing labels"

            # walk back from the last axis on each array to get the
            # correct labels/ticks
            these_axes = self.axes[:]
            those_axes = other.axes[:]
            ax_spec = []
            while these_axes and those_axes:
                this_ax = these_axes.pop(-1)
                that_ax = those_axes.pop(-1)
                if this_ax.label is None:
                    ax_spec.append(that_ax)
                else:
                    ax_spec.append(this_ax)
            ax_spec = ax_spec[::-1]
            # if the axes are not totally consumed on one array or the other,
            # then grab those labels/ticks for the rest of the dims
            if these_axes:
                ax_spec = these_axes + ax_spec
            elif those_axes:
                ax_spec = those_axes + ax_spec
        else:
            ax_spec = self.axes[:]

        res = obj.view(type(self))
        new_axes = []
        for i, ax in enumerate(ax_spec):
            new_axes.append( Axis(ax.label, i, res, ticks=ax.ticks) )
        _set_axes(res, new_axes)
        return res
        
    def transpose(self, *axes):
        """ accept integer or named axes, reorder axes """
        # implement tuple-or-*args logic of np.transpose
        axes = list(axes)
        if not axes:
            axes = range(self.ndim-1,-1,-1)
        # expand sequence if sequence passed as first and only arg
        elif len(axes) == 1:
            try:
                axes = axes[0][:]
            except TypeError:
                pass
            stop
        proc_axids = _names_to_numbers(self.axes, axes)
        out = np.ndarray.transpose(self, proc_axids)
        _set_axes(out, _reordered_axes(self.axes, proc_axids, parent=out))
        return out

    def __getitem__(self, key):
        """Support x[k] access."""
        # Slicing keys:
        # * a single int
        # * a single newaxis
        # * a tuple with length <= self.ndim (may have newaxes)
        # * a tuple with length > self.ndim (MUST have newaxes)
        # * list, array, etc for fancy indexing (not implemented)

        # Prickly points..
        # * When slicing causes a None-labeled dimension to
        #   change name, say from _1 to _0, but the list of
        #   names set up in this method remains static

        
        # Cases
        if isinstance(key, list) or isinstance(key, np.ndarray):
            # fancy indexing
            # XXX need to be cast to an "ordinary" ndarray
            raise NotImplementedError
        if key is None:
            key = (key,)

        if isinstance(key, tuple):
            old_shape = self.shape
            old_axes = self.axes[:]
            new_shape, new_axes, key = _make_singleton_axes(self, key)
            # Will undo this later
            self.shape = new_shape
            _set_axes(self, new_axes)
            # data is accessed recursively, starting with
            # the full array
            arr = self

            # We must copy of the names of the axes
            # before looping through the elements of key,
            # as the index of a given axis may change.
            names = [a.name for a in self.axes]

            # If an Axis gets sliced out entirely, then any following
            # unlabeled Axis in the array will spontaneously change name.
            # So anticipate the name change here.
            reduction = 0
            adjustments = []
            for k in key:
                adjustments.append(reduction)
                if not isinstance(k, slice):
                    # reduce the idx # on the remaining default labels
                    reduction -= 1

            names = [n if a.label else '_%d'%(a.index+r)
                     for n, a, r in zip(names, self.axes, adjustments)]

            for slice_or_int, name in zip(key, names):
                arr = arr.axis[name][slice_or_int]

            # restore old shape and axes
            self.shape = old_shape
            _set_axes(self, old_axes)
        else:
            arr = self.axes[0][key]

        return arr

    def __str__(self):
        s = super(DataArray, self).__str__()
        s = '\n'.join([s, str(self.labels)])
        return s

    def __repr__(self):
        s = super(DataArray, self).__repr__()
        s = '\n'.join([s, str(self.labels)])
        return s
    
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
        new_ax = ax.__class__(ax.label, new_ind, parent_arr, ticks=ax.ticks)
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
    with an unlabeled Axis at each singleton dimension; and the new
    slicing key, with `newaxis` keys replaced by slice(None)
    """
    
    key = _expand_ellipsis(key, arr.ndim)
    if len(key) <= arr.ndim and None not in key:
        return arr.shape, arr.axes[:], key

    # The full slicer will be length=arr.ndim + # of dummy-dims..
    # Boost up the slices to full "rank" ( can cut it down later for savings )
    n_new_dims = len(filter(lambda x: x is None, key))
    key = key + (slice(None),) * (arr.ndim + n_new_dims - len(key))
    # wherever there is a None in the key,
    # * replace it with slice(None)
    # * place a new dimension with length 1 in the shape,
    # * and add a new unlabeled Axis to the axes
    new_dims = []
    new_key = []
    d_cnt = 0
    new_ax_pos = arr.ndim
    new_axes = arr.axes[:]
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
    # it seems we have to leave in at least one slicing element
    # in order to get a new array
    while len(new_key)>1 and new_key[-1] == slice(None):
        new_key.pop()
    return tuple(new_dims), ro_axes, tuple(new_key)
    
    

    
