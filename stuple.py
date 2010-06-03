## __all__ = ['stuple']
from copy import copy

class stuple(list):

    def __init__(self, tup, this_axis=None, axes=[]):
        list.__init__(self, tup)
        self._axis = this_axis
        self._sub_axes = axes
        for ax in axes:
            ax_stuple = proxy_stuple(tup, this_axis=ax, axes=axes)
            setattr(self, ax.label, ax_stuple)

    def __getitem__(self, key):
        if not len(self):
            slicer = [ slice(None) ] * len(self._sub_axes)
        else:
            slicer = list(self)
        if not self._axis:
            raise IndexError("This stuple cannot be indexed")
        if slicer[self._axis.index] != slice(None):
            raise ValueError("This stuple has already been sliced")
        slicer[self._axis.index] = key
        return stuple(slicer, axes=self._sub_axes)

    def __repr__(self):
        return str(list(self))

    __str__ = __repr__
    
class proxy_stuple(object):

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __getitem__(self, key):
        stup = stuple(*self._args, **self._kwargs)
        print stup._axis, key
        return stup[key]


if __name__=='__main__':
    from datarray import Axis, DataArray
    import numpy as np
    x = np.random.randn(4,6,2,3)
    a = DataArray(x, labels=('x', 'y', 't', 'f'))
    s_anon = stuple( ( slice(None), ) * 4, axes=a.axes)
    print s_anon.x[:2].y[4:]
    y = x[ s_anon.x[:2].y[4:] ]
    print y, y.shape
