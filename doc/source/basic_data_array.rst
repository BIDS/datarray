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

    >>> narr = DataArray(np.zeros((1,2,3)), axes=('a', 'b', 'c'))
    >>> narr.names
    ('a', 'b', 'c')
    >>> narr.axis.a
    Axis(name='a', index=0, labels=None)
    >>> narr.axis.b
    Axis(name='b', index=1, labels=None)
    >>> narr.axis.c
    Axis(name='c', index=2, labels=None)
    >>> narr.shape
    (1, 2, 3)

Not all axes must necessarily be explicitly named, since None is a valid axis
name:

    >>> narr2 = DataArray(np.zeros((1,2,3)), axes=('a', None, 'b' ))
    >>> narr2.names
    ('a', None, 'b')

If no name is given for an axis, None is implicitly assumed.  So trailing axes
without axes will be named as None:

    >>> narr2 = DataArray(np.zeros((1,2,3,2)), axes=('a','b' ))
    >>> narr2.names
    ('a', 'b', None, None)

Combining named and unnamed arrays:

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
:

    >>> n4 = DataArray(np.ones((2,1,3)), axes=('b','a','c'))
    >>> res = narr + n4 # is this OK?
    Traceback (most recent call last):
    ...
    NamedAxisError: Axis names are incompatible for a binary operation: ('a', 'b', 'c'), ('b', 'a', 'c')

The names and the position have to be the same, and the above example should
raise an error.  At least for now we will raise an error, and review later.

With "labels"
------------

Constructing a DataArray such that an Axis has labels, for example:

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

    >>> narr = DataArray(np.zeros((1,2,3)), axes=('a','b','c'))
    >>> narr.axis.a
    Axis(name='a', index=0, labels=None)
    >>> narr.axis.a[0]
    DataArray([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])
    ('b', 'c')
    >>> narr.axis.a[0].axes
    (Axis(name='b', index=0, labels=None), Axis(name='c', index=1, labels=None))

By normal "numpy" slicing:

    >>> narr[0].shape
    (2, 3)
    >>> narr[0].axes
    (Axis(name='b', index=0, labels=None), Axis(name='c', index=1, labels=None))
    >>> narr.axis.a[0].axes == narr[0,:].axes
    True

Through the "axis slicer" ``aix`` attribute:

    >>> narr[ narr.aix.b[:2].c[-1] ]
    DataArray([[ 0.,  0.]])
    ('a', 'b')
    >>> narr[ narr.aix.c[-1].b[:2] ]
    DataArray([[ 0.,  0.]])
    ('a', 'b')
    >>> narr[ narr.aix.c[-1].b[:2] ] == narr[:,:2,-1]
    DataArray([[ True,  True]], dtype=bool)
    ('a', 'b')

The Axis Indexing object (it's a stuple)
----------------------------------------

The ``aix`` attribute is a property which generates a "stuple" (special/slicing tuple)::

    @property
    def aix(self):
        # Returns an anonymous slicing tuple that knows
        # about this array's geometry
        return stuple( ( slice(None), ) * self.ndim,
                       axes = self.axes )


The stuple should have a reference to a group of Axis objects that describes an
array's geometry. If the stuple is associated with a specific Axis, then when
sliced itself, it can create a slicing tuple for the array with the given
geometry.
:

    >>> narr.aix
    (slice(None, None, None), slice(None, None, None), slice(None, None, None))
    >>> narr.names
    ('a', 'b', 'c')
    >>> narr.aix.b[0]
    (slice(None, None, None), 0, slice(None, None, None))

**Note** -- the ``aix`` attribute provides some shorthand syntax for the following:

    >>> narr.axis.c[-1].axis.b[:2]
    DataArray([[ 0.,  0.]])
    ('a', 'b')

The mechanics are slightly different (using ``aix``, a slicing tuple is created
up-front before ``__getitem__`` is called), but functionality is the same.
**Question** -- Is it convenient enough to include the ``aix`` slicer? should
it function differently?

Also, slicing with ``newaxis`` is implemented:

    >>> arr = np.arange(24).reshape((3,2,4))
    >>> b = DataArray(arr, ['x', 'y', 'z'])
    >>> b[:,:,np.newaxis].shape
    (3, 2, 1, 4)
    >>> b[:,:,np.newaxis].names
    ('x', 'y', None, 'z')

I can also slice with ``newaxis`` at each Axis, or with the ``aix`` slicer (the
results are identical). The effect of this is always to insert an unnamed
Axis with length-1 at the original index of the named Axis:

    >>> b.axes
    (Axis(name='x', index=0, labels=None), Axis(name='y', index=1, labels=None), Axis(name='z', index=2, labels=None))
    >>> b.axis.y[np.newaxis].names
    ('x', None, 'y', 'z')
    >>> b.axis.y[np.newaxis].shape
    (3, 1, 2, 4)

Slicing and labels
-----------------

It is also possible to use labels in any of the slicing syntax above:

.. doctest::

    >>> time_caps #doctest: +NORMALIZE_WHITESPACE
    DataArray([[ 0,  1,  2,  3,  4],
     [ 5,  6,  7,  8,  9],
     [10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]])
    ('time', 'capitals')
    >>> time_caps.axis.capitals['berlin'::-1] #doctest: +NORMALIZE_WHITESPACE
    DataArray([[ 2,  1,  0],
     [ 7,  6,  5],
     [12, 11, 10],
     [17, 16, 15]])
    ('time', 'capitals')
    >>> time_caps.axis.time['0015':'1815'] #doctest: +NORMALIZE_WHITESPACE
    DataArray([[ 0,  1,  2,  3,  4],
     [ 5,  6,  7,  8,  9],
     [10, 11, 12, 13, 14]])
    ('time', 'capitals')
    >>> time_caps[:, 'london':3] #doctest: +NORMALIZE_WHITESPACE
    DataArray([[ 1,  2],
     [ 6,  7],
     [11, 12],
     [16, 17]])
    ('time', 'capitals')


The .start and .stop attributes of the slice object can be either None, an
integer index, or a valid tick. They may even be mixed. *The .step attribute,
however, must be None or an nonzero integer.*

**Historical note: previously integer labels clobbered indices.** For example::

    >>> centered_data = DataArray(np.random.randn(6), [ ('c_idx', range(-3,3)) ])
    >>> centered_data.axis.c_idx.make_slice( slice(0, 6, None) )
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

    >>> centered_data.axis.c_idx[ [0]:[2] ]
    >>> < returns underlying array from [3:5] >
    >>> centered_data.axis.c_idx[ 0:2 ]
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
    DataArray([[ 1.,  1.,  1.],
     [ 1.,  1.,  1.],
     [ 1.,  1.,  1.]])
    ('x', 'y')

When there are unnamed dimensions, they also must be consistently oriented
across arrays when broadcasting:

    >>> b = DataArray(np.arange(24).reshape(3,2,4), ['x', None, 'y'])
    >>> a = DataArray(np.arange(8).reshape(2,4), [None, 'y'])
    >>> res = a + b
    >>> res
    DataArray([[[ 0,  2,  4,  6],
	    [ 8, 10, 12, 14]],
    <BLANKLINE>
	   [[ 8, 10, 12, 14],
	    [16, 18, 20, 22]],
    <BLANKLINE>
	   [[16, 18, 20, 22],
	    [24, 26, 28, 30]]])
    ('x', None, 'y')

We already know that if the dimension names don't match, this won't be allowed (even though the shapes are correct):

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
    DataArray([[ 1.,  1.,  1.],
     [ 1.,  1.,  1.],
     [ 1.,  1.,  1.]])
    ('x', 'y')

In other words, this scenario is also a legal combination:

.. doctest::

    >>> a2 = a[:,None]
    >>> a2.names
    ('x', None)
    >>> b + a2    # doctest: +NORMALIZE_WHITESPACE
    DataArray([[ 2.,  2.,  2.],
     [ 2.,  2.,  2.],
     [ 2.,  2.,  2.]])
    ('x', 'y')

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
    DataArray([[ 2.,  2.,  2.],
     [ 2.,  2.,  2.],
     [ 2.,  2.,  2.]])
    ('x', 'y')

The broadcasting rules currently allow this combination. I'm inclined to allow
it. Even though the axes are different lengths in ``a`` and ``b``, and
therefore *might* be considered different logical axes, there is no actual
information collision from ``a.axis.y``.

.. _iteration:

Iteration
=========

seems to work:

    >>> for foo in time_caps:
    ...     print foo
    ...     print foo.axes
    ...
    [0 1 2 3 4]
    ('capitals',)
    (Axis(name='capitals', index=0, labels=['washington', 'london', 'berlin', 'paris', 'moscow']),)
    [5 6 7 8 9]
    ('capitals',)
    (Axis(name='capitals', index=0, labels=['washington', 'london', 'berlin', 'paris', 'moscow']),)
    [10 11 12 13 14]
    ('capitals',)
    (Axis(name='capitals', index=0, labels=['washington', 'london', 'berlin', 'paris', 'moscow']),)
    [15 16 17 18 19]
    ('capitals',)
    (Axis(name='capitals', index=0, labels=['washington', 'london', 'berlin', 'paris', 'moscow']),)

    >>> for foo in time_caps.T:
    ...    print foo
    ...    print foo.axes
    ...
    [ 0  5 10 15]
    ('time',)
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']),)
    [ 1  6 11 16]
    ('time',)
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']),)
    [ 2  7 12 17]
    ('time',)
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']),)
    [ 3  8 13 18]
    ('time',)
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']),)
    [ 4  9 14 19]
    ('time',)
    (Axis(name='time', index=0, labels=['0015', '0615', '1215', '1815']),)

Or even more conveniently:

    >>> for foo in time_caps.axis.capitals:
    ...     print foo
    ...
    [ 0  5 10 15]
    ('time',)
    [ 1  6 11 16]
    ('time',)
    [ 2  7 12 17]
    ('time',)
    [ 3  8 13 18]
    ('time',)
    [ 4  9 14 19]
    ('time',)

.. _transposition:

Transposition of Axes
=====================

Transposition of a DataArray preserves the dimension names, and updates the
corresponding indices:

    >>> b = DataArray(np.zeros((3, 2, 4)), axes=['x', None, 'y'])
    >>> b.shape
    (3, 2, 4)
    >>> b.axes
    (Axis(name='x', index=0, labels=None), Axis(name=None, index=1, labels=None), Axis(name='y', index=2, labels=None))
    >>> b.T.shape
    (4, 2, 3)
    >>> b.T.axes
    (Axis(name='y', index=0, labels=None), Axis(name=None, index=1, labels=None), Axis(name='x', index=2, labels=None))

.. _label_updates:

Changing Names on DataArrays
=============================

Tricky Attributes
-----------------

* .names -- currently a mutable list of Axis.name attributes
* .axes -- currently a mutable list of Axis objects
* .axis -- a key-to-attribute dictionary

Need an event-ful way to change an Axis's label, such that all the above
attributes are updated.

**Proposed solution**: 

1. use a set_label() method. This will consequently update the parent array's 
    (names, axes, axis) attributes. 
2. make the mutable lists into *tuples* to deny write access.
3. make the KeyStruct ``.axis`` have write-once access 

.. _todo:

ToDo
====

* Support DataArray instances with mixed axes: simple ones with no values 
  and 'fancy' ones with data in them.  Syntax?

``a = DataArray.from_names(data, axes=['a','b','c'])``

``b = DataArray(data, axes=[('a',['1','2','3']), ('b',['one','two']), ('c',['red','black'])])``

``c = DataArray(data, axes=[('a',['1','2','3']), ('b',None), ('c',['red','black'])])``

* Can a, b, and c be combined in binary operations, given the different tick
  combinations?
* How to handle complicated reshaping (not flattening or, padding/trimming with
  1s) 
* Units support (Darren's)
* Jagged arrays? Kilian's suggestion.  Drop the base array altogether, and
  access data via the .axis objects alone.
* "Enum dtype", could be useful for event selection.
* "Ordered factors"? Something R supports.
* How many axis classes?

* Allowing non-string axis names?

- At least they must be hashable...
- Serialization?


* Allowing multiple names per axis?


* Rob Speer's proposal for purely top-level, 'magical' attributes?


* Finish the semantics of .lix indexing, especially with regards to what it
  should do when integer labels are present.

* What should a.axis.x[object] do: .lix-style indexing or pure numpy indexing?

Indexing semantics possibilities
--------------------------------

1. .lix: Integers always labels.  a.lix[3:10] means labels 3 and 10 MUST exist.

2. .nix: Integers are never treated as labels.

3. .awful_ix: 1, then 2.


Axis api
--------
If a is an axis from an array: a = x.axis.a

- a.at(key): return the slice at that key, with one less dimension than x
- a.keep(keys): join slices for given keys, dims=dims(x)
- a.drop(keys): like keep, but the opposite

a[i] valid cases:

- i: integer => normal numpy scalar indexing, one less dim than x
- i: slice: numpy view slicing.  same dims as x, must recover the labels 
- i: list/array: numpy fancy indexing, as long as the index list is 1d only.

