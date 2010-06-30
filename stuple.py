__all__ = ['stuple', 'StupleSlicingError']

class StupleSlicingError(Exception):
    """Class for slicing/indexing exceptions in a stuple"""
    pass

class stuple(tuple):
    """A special tuple? A slicing tuple? All of the above?

    A stuple has a reference to a list of Axis objects, and may be
    associated with a particular Axis. For each valid label in the
    axes list, that label becomes an attribute on the stuple, to
    allow chaining together slicing on multiple axes.

    Example
    -------
    
    >>> s = stuple.stuple( (), axes=[ Axis(label, idx, None) for label, idx in zip('abc', [0,1,2]) ] )
    >>> hasattr(s, 'a')
    True
    >>> hasattr(s, 'b')
    True
    >>> s.a[0:2]
    (slice(0, 2, None), slice(None, None, None), slice(None, None, None))
    >>> s.a[0:2].c[::2]
    (slice(0, 2, None), slice(None, None, None), slice(None, None, 2))
    >>> s
    ()

    
    """

    def __new__(klass, tup, this_axis=None, axes=[]):
        t = tuple.__new__(klass, tup)
        t._axis = this_axis
        t._all_axes = axes
        for ax in axes:
            # put some embryonic stuples on ice until they are accessed
            # by name and sliced (see proxy_stuple.__getitem__)
            # (This is to avoid infinite recursion)
            if ax.label is not None:
                ax_stuple = proxy_stuple(tup, this_axis=ax, axes=axes)
                setattr(t, ax.label, ax_stuple)
        return t

    def __getitem__(self, key):
        # getitem has some subtle behavior..

        # In any case, list-ify the self-sequence to make it mutable

        # 1st case allows a construction of a top-level
        # stuple with no explicit length: stuple( (), axes=arr.axes ).
        if not len(self):
            slicer = [ slice(None) ] * len(self._all_axes)
        else:
            slicer = list(self)

        # Now, there are two modes of access.
        # If this stuple has no associated axis, then slice it like a tuple
        if self._axis is None:
            return super(stuple, self).__getitem__(key)

        # Otherwise, slice it to return a slicing tuple (stuple)

        # However, once a stuple has been sliced along its axis, it
        # cannot be sliced again.
        if slicer[self._axis.index] != slice(None):
            raise StupleSlicingError("This axis has already been sliced")
        
        slicer[self._axis.index] = key
        return stuple(slicer, axes=self._all_axes)

    def __repr__(self):
        return str(tuple(self))

    __str__ = __repr__
    
class proxy_stuple(object):
    "A stuple frozen in a pre-instantiated state until it is sliced"
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __getitem__(self, key):
        stup = stuple(*self._args, **self._kwargs)
        return stup[key]

