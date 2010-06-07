__all__ = ['stuple', 'StupleSlicingError']
from copy import copy

class StupleSlicingError(Exception):
    """Class for slicing/indexing exceptions in a stuple"""
    pass

class stuple(tuple):

    def __new__(klass, tup, this_axis=None, axes=[]):
        t = tuple.__new__(klass, tup)
        t._axis = this_axis
        t._all_axes = axes
        for ax in axes:
            ax_stuple = proxy_stuple(tup, this_axis=ax, axes=axes)
            setattr(t, ax.label, ax_stuple)
        return t

    def __getitem__(self, key):
        if not len(self):
            slicer = [ slice(None) ] * len(self._all_axes)
        else:
            slicer = list(self)
        if not self._axis:
            raise StupleSlicingError("This stuple cannot be indexed")
        if slicer[self._axis.index] != slice(None):
            raise StupleSlicingError("This axis has already been sliced")
        slicer[self._axis.index] = key
        return stuple(slicer, axes=self._all_axes)

    def __repr__(self):
        return str(tuple(self))

    __str__ = __repr__
    
class proxy_stuple(object):

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __getitem__(self, key):
        stup = stuple(*self._args, **self._kwargs)
        return stup[key]

