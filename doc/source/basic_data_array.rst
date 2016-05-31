.. testsetup::

    import numpy as np
    from datarray import DataArray

============
 DataArrays
============

.. _init_ufuncs:


Basic DataArray Creation And Mixing
===================================

DataArrays are constructed with array-like sequences and axis names:

.. doctest::

    >>> narr = DataArray(np.zeros((1,2,3)), axes=('a', 'b', 'c'))
    >>> narr.names
    ('a', 'b', 'c')
    >>> narr.axes.a
    Axis(name='a', index=0, labels=None)
    >>> narr.axes.b
    Axis(name='b', index=1, labels=None)
    >>> narr.axes.c
    Axis(name='c', index=2, labels=None)
    >>> narr.shape
    (1, 2, 3)

Not all axes must necessarily be explicitly named, since None is a valid axis
name:

.. doctest::

    >>> narr2 = DataArray(np.zeros((1,2,3)), axes=('a', None, 'b' ))
    >>> narr2.names
    ('a', None, 'b')

If no name is given for an axis, None is implicitly assumed.  So trailing axes
without axes will be named as None:

.. doctest::

    >>> narr2 = DataArray(np.zeros((1,2,3,2)), axes=('a','b' ))
    >>> narr2.names
    ('a', 'b', None, None)

Combining named and unnamed arrays:

.. doctest::

    >>> narr = DataArray(np.zeros((1,2,3)), axes='abc')
    >>> res = narr + 5 # OK
    >>> res = narr + np.zeros((1,2,3)) # OK
    >>> n2 = DataArray(np.ones((1,2,3)), axes=('a','b','c'))
    >>> res = narr + n2 # OK

    >>> n3 = DataArray(np.ones((1,2,3)), axes=('x','b','c'))

    >>> res = narr + n3
    Traceback (most recent call last):
    ...
    NamedAxisError: Axis names are incompatible for a binary operation: ('a', 'b', 'c'), ('x', 'b', 'c')


Now, what about matching names, but different indices for the names?

.. doctest::

    >>> n4 = DataArray(np.ones((2,1,3)), axes=('b','a','c'))
    >>> res = narr + n4 # is this OK?
    Traceback (most recent call last):
    ...
    NamedAxisError: Axis names are incompatible for a binary operation: ('a', 'b', 'c'), ('b', 'a', 'c')

The names and the position have to be the same, and the above example should
raise an error.  At least for now we will raise an error, and review later.

With "labels"
-------------

Constructing a DataArray such that an Axis has labels, for example:

.. doctest::

    >>> cap_ax_spec = 'capitals', ['washington', 'london', 'berlin', 'paris', 'moscow']
    >>> time_ax_spec = 'time', ['0015', '0615', '1215', '1815']
    >>> time_caps = DataArray(np.arange(4*5).reshape(4,5), [time_ax_spec, cap_ax_spec])
    >>> time_caps.axes
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']), Axis(name='capitals', index=1, labels=['washington', 'london', 'berlin', 'paris', 'moscow']))

.. _slicing:

Slicing
=======

A DataArray with simple named axes can be sliced many ways.

Per Axis:

.. doctest::

    >>> narr = DataArray(np.zeros((1,2,3)), axes=('a','b','c'))
    >>> narr.axes.a
    Axis(name='a', index=0, labels=None)
    >>> narr.axes.a[0]
    DataArray(array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]]),
    ('b', 'c'))
    >>> narr.axes.a[0].axes
    (Axis(name='b', index=0, labels=None), Axis(name='c', index=1, labels=None))

By normal "numpy" slicing:

.. doctest::

    >>> narr[0].shape
    (2, 3)
    >>> narr[0].axes
    (Axis(name='b', index=0, labels=None), Axis(name='c', index=1, labels=None))
    >>> narr.axes.a[0].axes == narr[0,:].axes
    True

Also, slicing with ``newaxis`` is implemented:

.. doctest::

    >>> arr = np.arange(24).reshape((3,2,4))
    >>> b = DataArray(arr, ['x', 'y', 'z'])
    >>> b[:,:,np.newaxis].shape
    (3, 2, 1, 4)
    >>> b[:,:,np.newaxis].names
    ('x', 'y', None, 'z')

I can also slice with ``newaxis`` at each Axis.  The effect of this is always
to insert an unnamed Axis with length-1 at the original index of the named
Axis:

.. doctest::

    >>> b.axes
    (Axis(name='x', index=0, labels=None), Axis(name='y', index=1, labels=None), Axis(name='z', index=2, labels=None))
    >>> b.axes.y[np.newaxis].names
    ('x', None, 'y', 'z')
    >>> b.axes.y[np.newaxis].shape
    (3, 1, 2, 4)

Slicing and labels
------------------

It is also possible to use labels in any of the slicing syntax above:

.. doctest::

    >>> time_caps #doctest: +NORMALIZE_WHITESPACE
    DataArray(array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]]),
    (('time', ('0015', '0615', '1215', '1815')), ('capitals', ('washington', 'london', 'berlin', 'paris', 'moscow'))))
    >>> time_caps.axes.capitals['berlin'::-1] #doctest: +NORMALIZE_WHITESPACE
    DataArray(array([[ 2,  1,  0],
           [ 7,  6,  5],
           [12, 11, 10],
           [17, 16, 15]]),
    (('time', ('0015', '0615', '1215', '1815')), ('capitals', ('berlin', 'london', 'washington'))))
    >>> time_caps.axes.time['0015':'1815'] #doctest: +NORMALIZE_WHITESPACE
    DataArray(array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]]),
    (('time', ('0015', '0615', '1215')), ('capitals', ('washington', 'london', 'berlin', 'paris', 'moscow'))))
    >>> time_caps[:, 'london':3] #doctest: +NORMALIZE_WHITESPACE
    DataArray(array([[ 1,  2],
           [ 6,  7],
           [11, 12],
           [16, 17]]),
    (('time', ('0015', '0615', '1215', '1815')), ('capitals', ('london', 'berlin'))))

The .start and .stop attributes of the slice object can be either None, an
integer index, or a valid tick. They may even be mixed. *The .step attribute,
however, must be None or an nonzero integer.*

**Historical note: previously integer labels clobbered indices.** For example::

    >>> centered_data = DataArray(np.random.randn(6), [ ('c_idx', range(-3,3)) ])
    >>> centered_data.axes.c_idx.make_slice( slice(0, 6, None) )
    (slice(3, 6, None),)

.. note::

   The code above doesn't currently (as of Nov/2010) run, because integer
   labels haven't been implemented.  See ticket gh-40.
    
make_slice() first tries to look up the key parameters as labels, and then sees
if the key parameters can be used as simple indices. Thus 0 is found as index
3, and 6 is passed through as index 6.

Possible resolution 1
~~~~~~~~~~~~~~~~~~~~~

"larry" would make this distinction::

    >>> centered_data.axes.c_idx[ [0]:[2] ]
    >>> < returns underlying array from [3:5] >
    >>> centered_data.axes.c_idx[ 0:2 ]
    >>> < returns underlying array from [0:2] >

And I believe mixing of labels and is valid also.

Possible resolution 2 (the winner)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do not allow integer labels -- cast to float perhaps

**Note**: this will be the solution. When validating labels on an Axis, ensure
that none of them ``isinstance(t, int)``


Possible resolution 3
~~~~~~~~~~~~~~~~~~~~~

Restrict access to tick based slicing to another special slicing object.

.. _broadcasting:

Broadcasting
============

What about broadcasting between two named arrays, where the broadcasting
adds an axis? All ordinary NumPy rules for shape compatibility apply.
Additionally, DataArray imposes axis name consistency rules.

The broadcasted DataArray below, "a", takes on dummy dimensions that are taken
to be compatible with the larger DataArray:

.. doctest::

    >>> b = DataArray(np.ones((3,3)), axes=('x','y'))
    >>> a = DataArray(np.ones((3,)), axes=('y',))
    >>> res = 2*b - a
    >>> res    # doctest: +NORMALIZE_WHITESPACE
    DataArray(array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 1.,  1.,  1.]]),
    ('x', 'y'))

When there are unnamed dimensions, they also must be consistently oriented
across arrays when broadcasting:

.. doctest::

    >>> b = DataArray(np.arange(24).reshape(3,2,4), ['x', None, 'y'])
    >>> a = DataArray(np.arange(8).reshape(2,4), [None, 'y'])
    >>> res = a + b
    >>> res
    DataArray(array([[[ 0,  2,  4,  6],
            [ 8, 10, 12, 14]],
    <BLANKLINE>
           [[ 8, 10, 12, 14],
            [16, 18, 20, 22]],
    <BLANKLINE>
           [[16, 18, 20, 22],
            [24, 26, 28, 30]]]),
    ('x', None, 'y'))

We already know that if the dimension names don't match, this won't be allowed
(even though the shapes are correct):

.. doctest::

    >>> b = DataArray(np.ones((3,3)), axes=('x','y'))
    >>> a = DataArray(np.ones((3,)), axes=('x',))
    >>> res = 4*b - a
    Traceback (most recent call last):
    ...
    NamedAxisError: Axis names are incompatible for a binary operation: ('x', 'y'), ('x',)

But a numpy idiom for padding dimensions helps us in this case:

.. doctest::

    >>> res = 2*b - a[:,None]
    >>> res    # doctest: +NORMALIZE_WHITESPACE
    DataArray(array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 1.,  1.,  1.]]),
    ('x', 'y'))

In other words, this scenario is also a legal combination:

.. doctest::

    >>> a2 = a[:,None]
    >>> a2.names
    ('x', None)
    >>> b + a2    # doctest: +NORMALIZE_WHITESPACE
    DataArray(array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.],
           [ 2.,  2.,  2.]]),
    ('x', 'y'))

The rule for dimension compatibility is that any two axes match if one of the following is true

* their (name, length) pairs are equal
* their dimensions are broadcast-compatible, and their axes are equal
* their dimensions are broadcast-compatible, and their axes are
  non-conflicting (ie, one or both are None)

**Question** -- what about this situation:

.. doctest::

    >>> b = DataArray(np.ones((3,3)), axes=('x','y'))
    >>> a = DataArray(np.ones((3,1)), axes=('x','y'))
    >>> a+b          # doctest: +NORMALIZE_WHITESPACE
    DataArray(array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.],
           [ 2.,  2.,  2.]]),
    ('x', 'y'))

The broadcasting rules currently allow this combination. I'm inclined to allow
it. Even though the axes are different lengths in ``a`` and ``b``, and
therefore *might* be considered different logical axes, there is no actual
information collision from ``a.axes.y``.

.. _iteration:

Iteration
=========

seems to work:

.. doctest::

    >>> for foo in time_caps:
    ...     print foo
    ...     print foo.axes
    ...
    DataArray([0 1 2 3 4],
    (('capitals', ('washington', 'london', 'berlin', 'paris', 'moscow')),))
    (Axis(name='capitals', index=0, labels=['washington', 'london', 'berlin', 'paris', 'moscow']),)
    DataArray([5 6 7 8 9],
    (('capitals', ('washington', 'london', 'berlin', 'paris', 'moscow')),))
    (Axis(name='capitals', index=0, labels=['washington', 'london', 'berlin', 'paris', 'moscow']),)
    DataArray([10 11 12 13 14],
    (('capitals', ('washington', 'london', 'berlin', 'paris', 'moscow')),))
    (Axis(name='capitals', index=0, labels=['washington', 'london', 'berlin', 'paris', 'moscow']),)
    DataArray([15 16 17 18 19],
    (('capitals', ('washington', 'london', 'berlin', 'paris', 'moscow')),))
    (Axis(name='capitals', index=0, labels=['washington', 'london', 'berlin', 'paris', 'moscow']),)

    >>> for foo in time_caps.T:
    ...    print foo
    ...    print foo.axes
    ...
    DataArray([ 0  5 10 15],
    (('time', ('0015', '0615', '1215', '1815')),))
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']),)
    DataArray([ 1  6 11 16],
    (('time', ('0015', '0615', '1215', '1815')),))
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']),)
    DataArray([ 2  7 12 17],
    (('time', ('0015', '0615', '1215', '1815')),))
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']),)
    DataArray([ 3  8 13 18],
    (('time', ('0015', '0615', '1215', '1815')),))
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']),)
    DataArray([ 4  9 14 19],
    (('time', ('0015', '0615', '1215', '1815')),))
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']),)

Or even more conveniently:

.. doctest::

    >>> for foo in time_caps.axes.capitals:
    ...     print foo
    ...
    DataArray([ 0  5 10 15],
    (('time', ('0015', '0615', '1215', '1815')),))
    DataArray([ 1  6 11 16],
    (('time', ('0015', '0615', '1215', '1815')),))
    DataArray([ 2  7 12 17],
    (('time', ('0015', '0615', '1215', '1815')),))
    DataArray([ 3  8 13 18],
    (('time', ('0015', '0615', '1215', '1815')),))
    DataArray([ 4  9 14 19],
    (('time', ('0015', '0615', '1215', '1815')),))

.. _transposition:

Transposition of Axes
=====================

Transposition of a DataArray preserves the dimension names, and updates the
corresponding indices:

.. doctest::

    >>> b = DataArray(np.zeros((3, 2, 4)), axes=['x', None, 'y'])
    >>> b.shape
    (3, 2, 4)
    >>> b.axes
    (Axis(name='x', index=0, labels=None), Axis(name=None, index=1, labels=None), Axis(name='y', index=2, labels=None))
    >>> b.T.shape
    (4, 2, 3)
    >>> b.T.axes
    (Axis(name='y', index=0, labels=None), Axis(name=None, index=1, labels=None), Axis(name='x', index=2, labels=None))

