#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import copy

import numpy as np
import nose.tools as nt

from stuple import stuple

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


class Axis(object):
    """Object to access a given axis of an array.

    Key point: every axis contains a  reference to its parent array!
    """
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

        >>> a1 = Axis('time', 0, None, labels=[str(i) for i in xrange(10)])
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
        if labels and len(labels) != len(self.labels):
            ax._label_dict = dict( zip(labels, xrange( len(labels) )) )
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
        t_dict = dict(zip(labels, xrange(nlabels)))
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
        newaxes = [pa.axes[i] for i in xrange(self.index)]
        newaxes += [self]
        newaxes += [pa.axes[i] for i in xrange(self.index+1,nd)]
        _set_axes(pa, newaxes)
        
    def __len__(self):
        return self.parent_arr.shape[self.index]

    def __eq__(self, other):
        '''
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
        '''
        return self.name == other.name and self.index == other.index and \
               self.labels == other.labels

    def __str__(self):
        return 'Axis(name=%r, index=%i, labels=%r)' % \
               (self.name, self.index, self.labels)

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
        >>> arr = narr.axis.b.at('c')
        >>> arr.axes
        (Axis(name='a', index=0, labels=None),)
        >>>     

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
        >>> arr = narr.axis.b.keep('cd')
        >>> [a.labels for a in arr.axes]
        [None, 'cd']
        
        >>> arr.axis.a.at('label')
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
        >>> arr1 = darr.axis.b.keep(['c','d'])
        >>> arr2 = darr.axis.b.drop(['a','b','e'])
        >>> np.alltrue(np.equal(arr1, arr2))
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



def _validate_axes(axes):
    """
    This should always be true our axis lists....
    """
    p = axes[0].parent_arr
    for i, a in enumerate(axes):
        nt.assert_equals(i, a.index)
        nt.assert_true(p is a.parent_arr)

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
    """Set the axes in `dest` from `in_axes`.

    WARNING: The destination is modified in-place!  The following attributes
    are added to it:

    - axis: a KeyStruct with each axis as a named attribute.
    - axes: a list of all axis instances.

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
            raise NamedAxisError(
                'There is another Axis in this group with ' \
                'the same name'
                )
        ax_holder[ax._sname] = new_ax
    # Store these containers as attributes of the destination array
    dest.axes = tuple(axes)
    dest.axis = ax_holder
    

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
            return super_op(inst, **kwargs)

        axes = list(inst.axes)
        # try to convert a named Axis to an integer..
        # don't try to catch an error
        axis_idx = _names_to_numbers(inst.axes, [axis])[0]
        axes = _pull_axis(axes, inst.axes[axis_idx])
        kwargs['axis'] = axis_idx
        arr = super_op(inst, **kwargs)
        if not is_numpy_scalar(arr): 
            _set_axes(arr, axes)
        return arr
    runs_op.func_name = opname
    runs_op.func_doc = super_op.__doc__
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
    runs_op.func_name = opname
    runs_op.func_doc = super_op.__doc__
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
            raise NamedAxisError('axes list should have length <= array ndim')
        
        axes = list(axes) + [None]*(arr.ndim - len(axes))
        axlist = []
        for i, axis_spec in enumerate(axes):
            if type(axis_spec) == tuple:
                if len(axis_spec) != 2:
                    raise ValueError(
                        'if the axis specification is a tuple, it must be ' \
                        'of the form (name, labels)'
                        )
                name, labels = axis_spec
            else:
                name = axis_spec
                labels = None
            axlist.append(Axis(name, i, arr, labels=labels))

        _set_axes(arr, axlist)
        _validate_axes(axlist)

        return arr

    @property
    def aix(self):
        # Returns an anonymous slicing tuple that knows
        # about this array's geometry
        return stuple( ( slice(None), ) * self.ndim,
                       axes = self.axes )

    def set_name(self, i, name):
        self.axes[i].set_name(name)

    @property
    def names (self):
        """Returns a tuple with all the axis names."""
        return tuple((ax.name for ax in self.axes))

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
        _validate_axes(self.axes)

    def __array_prepare__(self, obj, context=None):
        """Called at the beginning of each ufunc.
        """

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
        #print "prepare:\t%s\n\t\t%s" % (self.__class__, obj.__class__) # dbg
        #print "obj     :", obj.shape  # dbg
        #print "context :", context # dbg

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
            # data is accessed recursively, starting with
            # the full array
            arr = self

            # We must copy of the names of the axes
            # before looping through the elements of key,
            # as the index of a given axis may change.
            names = [a.name for a in self.axes]

            # If an Axis gets sliced out entirely, then any following
            # unnamed Axis in the array will spontaneously change name.
            # So anticipate the name change here.
            reduction = 0
            adjustments = []
            for k in key:
                adjustments.append(reduction)
                if not isinstance(k, slice):
                    # reduce the idx # on the remaining default names
                    reduction -= 1

            names = [n if a.name else '_%d'%(a.index+r)
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
        s = '\n'.join([s, str(self.names)])
        return s

    def __repr__(self):
        s = super(DataArray, self).__repr__()
        s = '\n'.join([s, str(self.names)])
        return s

    # Methods from ndarray

    def transpose(self, *axes):
        # implement tuple-or-*args logic of np.transpose
        axes = list(axes)
        if not axes:
            axes = range(self.ndim-1,-1,-1)
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
    transpose.func_doc = np.ndarray.transpose.__doc__

    def swapaxes(self, axis1, axis2):
        # form a transpose operation with axes specified
        # by (axis1, axis2) swapped
        axis1, axis2 = _names_to_numbers(self.axes, [axis1, axis2])
        ax_idx = range(self.ndim)
        tmp = ax_idx[axis1]
        ax_idx[axis1] = ax_idx[axis2]
        ax_idx[axis2] = tmp
        out = np.ndarray.transpose(self, ax_idx)
        _set_axes(out, _reordered_axes(self.axes, ax_idx, parent=out))
        return out
    swapaxes.func_doc = np.ndarray.swapaxes.__doc__

    def ptp(self, axis=None, out=None):
        mn = self.min(axis=axis)
        mx = self.max(axis=axis, out=out)
        if isinstance(mn, np.ndarray):
            mx -= mn
            return mx
        else:
            return mx-mn
    ptp.func_doc = np.ndarray.ptp.__doc__

    # -- Various extraction and reshaping methods ----------------------------
    def diagonal(self, *args, **kwargs):
        # reverts to being an ndarray
        args = (np.asarray(self),) + args
        return np.diagonal(*args, **kwargs)
    diagonal.func_doc = np.ndarray.diagonal.__doc__
    
    def flatten(self, **kwargs):
        # reverts to being an ndarray
        return np.asarray(self).flatten(**kwargs)
    flatten.func_doc = np.ndarray.flatten.__doc__

    def ravel(self, **kwargs):
        # reverts to being an ndarray
        return np.asarray(self).ravel(**kwargs)
    ravel.func_doc = np.ndarray.ravel.__doc__

    def repeat(self, *args, **kwargs):
        raise NotImplementedError

    def squeeze(self):
        axes = list(self.axes)
        pinched_axes = filter(lambda x: self.shape[x.index]==1, axes)
        squeezed_shape = filter(lambda d: d>1, self.shape)
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
        old_non_single_dims = filter(lambda d: d>1, old_shape)
        new_non_single_dims = filter(lambda d: d>1, new_shape)
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
            axes_to_pull += [self.axes[i] for i in xrange(j, len(old_shape))]
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
    mean = _apply_reduction('mean', ('axis', 'dtype', 'out'))
    var = _apply_reduction('var', ('axis', 'dtype', 'out', 'ddof'))
    std = _apply_reduction('std', ('axis', 'dtype', 'out', 'ddof'))

    min = _apply_reduction('min', ('axis', 'out'))
    max = _apply_reduction('max', ('axis', 'out'))

    sum = _apply_reduction('sum', ('axis', 'dtype', 'out'))
    prod = _apply_reduction('prod', ('axis', 'dtype', 'out'))
    
    ### these change the meaning of the axes..
    ### should probably return ndarrays
    argmax = _apply_reduction('argmax', ('axis',))
    argmin = _apply_reduction('argmin', ('axis',))

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
    n_new_dims = len(filter(lambda x: x is None, key))
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

