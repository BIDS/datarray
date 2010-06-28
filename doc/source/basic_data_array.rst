==========
DataArrays
==========

Questions
^^^^^^^^^

* :ref:`Constructing and Combining <init_ufuncs>`
* :ref:`Slicing <slicing>`
* :ref:`Broadcasting <broadcasting>`
* :ref:`Transposition, Rollaxes, Swapaxes <transposition>`
* :ref:`Iteration <iteration>`
* :ref:`Label Changing <label_updates>`
* :doc:`Wrapping functions with 'axis=' kw<ndarray_methods>`

.. _init_ufuncs:

Basic DataArray Creation And Mixing
-------------------------------------

DataArrays are constructed with array-like sequences and axis labels::

  >>> narr = DataArray(np.zeros((1,2,3)), labels=('a','b','c'))
  >>> narr.labels
  ['a', 'b', 'c']
  >>> narr.axis.a
  Axis(label='a', index=0, ticks=None)
  >>> narr.axis.b
  Axis(label='b', index=1, ticks=None)
  >>> narr.axis.c
  Axis(label='c', index=2, ticks=None)
  >>> narr.shape
  (1, 2, 3)

Not all axes must necessarily be explicitly labeled, since None is a valid axis
label::

  >>> narr2 = DataArray(np.zeros((1,2,3)), labels=('a', None, 'b' ))
  >>> narr2.labels
  ['a', None, 'b']

If no label is given for an axis, None is implicitly assumed.  So trailing axes
without labels will be labeled as None::

  >>> narr2 = DataArray(np.zeros((1,2,3,2)), labels=('a','b' ))
  >>> narr2.labels
  ['a', 'b', None, None]

With "ticks"
````````````

Constructing a DataArray such that an Axis has ticks, for example::

  >>> cap_ax_spec = 'capitals', ['washington', 'london', 'berlin', 'paris', 'moscow']
  >>> time_ax_spec = 'time', ['0015', '0615', '1215', '1815']
  >>> time_caps = DataArray(np.arange(4*5).reshape(4,5), [time_ax_spec, cap_ax_spec])
  >>> time_caps.axes
  [Axis(label='time', index=0, ticks=['0015', '0615', '1215', '1815']), Axis(label='capitals', index=1, ticks=['washington', 'london', 'berlin', 'paris', 'moscow'])]

Combining named and unnamed arrays::

  >>> res = narr + 5 # OK
  >>> res = narr + np.zeros((1,2,3)) # OK
  >>> n2 = DataArray(np.ones((1,2,3)), labels=('a','b','c'))
  >>> res = narr + n2 # OK

  >>> n3 = DataArray(np.ones((1,2,3)), labels=('x','b','c'))

  >>> res = narr + n3
  Traceback (most recent call last):
  ...
  NamedAxisError: Axis labels are incompatible for a binary operation: ['a', 'b', 'c'], ['x', 'b', 'c']

Now, what about matching names, but different indices for the names?
::

  >>> n4 = DataArray(np.ones((2,1,3)), labels=('b','a','c'))
  >>> res = narr + n4 # is this OK?
  Traceback (most recent call last):
  ...
  NamedAxisError: Axis labels are incompatible for a binary operation: ['a', 'b', 'c'], ['b', 'a', 'c']

The names and the position have to be the same, and the above example should raise an error.  At least for now we will raise an error, and review later.

.. _slicing:

Slicing
-------

A DataArray with simple named axes can be sliced many ways.

Per Axis::

  >>> narr = DataArray(np.zeros((1,2,3)), labels=('a','b','c'))
  >>> narr.axis.a
  Axis(label='a', index=0, ticks=None)
  >>> narr.axis.a[0]
  DataArray([[ 0.,  0.,  0.],
	 [ 0.,  0.,  0.]])
  >>> narr.axis.a[0].axes
  [Axis(label='b', index=0, ticks=None), Axis(label='c', index=1, ticks=None)]

By normal "numpy" slicing::

  >>> narr[0].shape
  (2, 3)
  >>> narr[0].axes
  [Axis(label='b', index=0, ticks=None), Axis(label='c', index=1, ticks=None)]
  >>> narr.axis.a[0].axes == narr[0,:].axes
  True

Through the "axis slicer" ``aix`` attribute::

  >>> narr[ narr.aix.b[:2].c[-1] ]
  DataArray([[ 0.,  0.]])
  >>> narr[ narr.aix.c[-1].b[:2] ]
  DataArray([[ 0.,  0.]])
  >>> narr[ narr.aix.c[-1].b[:2] ] == narr[:,:2,-1]
  DataArray([[ True,  True]], dtype=bool)

The Axis Indexing object (it's a stuple)
````````````````````````````````````````

The ``aix`` attribute is a property which generates a "stuple" (special/slicing tuple)::

    @property
    def aix(self):
        # Returns an anonymous slicing tuple that knows
        # about this array's geometry
        return stuple( ( slice(None), ) * self.ndim,
                       axes = self.axes )


The stuple should have a reference to a group of Axis objects that describes an array's geometry. If the stuple is associated with a specific Axis, then when sliced itself, it can create a slicing tuple for the array with the given geometry.
::

  >>> narr.aix
  (slice(None, None, None), slice(None, None, None), slice(None, None, None))
  >>> narr.labels
  ['a', 'b', 'c']
  >>> narr.aix.b[0]
  (slice(None, None, None), 0, slice(None, None, None))

**Note** -- the ``aix`` attribute provides some shorthand syntax for the following::

   >>> narr.axis.c[-1].axis.b[:2]
  DataArray([[ 0.,  0.]])

The mechanics are slightly different (using ``aix``, a slicing tuple is created up-front before ``__getitem__`` is called), but functionality is the same. **Question** -- Is it convenient enough to include the ``aix`` slicer? should it function differently?

Also, slicing with ``newaxis`` is implemented::

  >>> b = DataArray(np.random.randn(3,2,4), ['x', 'y', 'z'])
  >>> b[:,:,np.newaxis]
  >>> b[:,:,np.newaxis].shape
  (3, 2, 1, 4)
  >>> b[:,:,np.newaxis].labels
  ['x', 'y', None, 'z']

I can also slice with ``newaxis`` at each Axis, or with the ``aix`` slicer (the results are identical). The effect of this is always to insert an unlabeled Axis with length-1 at the original index of the named Axis::

  >>> b.axes
  [Axis(label='x', index=0, ticks=None), Axis(label='y', index=1, ticks=None), Axis(label='z', index=2, ticks=None)]
  >>> b.axis.y[np.newaxis]
  DataArray([[[[-0.5185789 ,  2.15360928,  0.27439545,  1.03371466],
	   [ 0.22295004, -0.67102797, -0.84618714, -0.87435244]]],


	 [[[ 1.22570705, -1.33283074, -0.89732455,  0.87430548],
	   [-0.69306908, -0.25327027, -0.53897745, -0.8659791 ]]],


	 [[[-1.18462101, -0.1644404 ,  0.5840826 ,  1.36768481],
	   [-0.51897418, -0.43526721, -1.18011399,  1.3553315 ]]]])
  ['x', None, 'y', 'z']
  >>> b.axis.y[np.newaxis].labels
  ['x', None, 'y', 'z']
  >>> b.axis.y[np.newaxis].shape
  (3, 1, 2, 4)

Slicing and ticks
`````````````````

It is also possible to use ticks in any of the slicing syntax above. 

::

  >>> time_caps
  DataArray([[ 0,  1,  2,  3,  4],
	 [ 5,  6,  7,  8,  9],
	 [10, 11, 12, 13, 14],
	 [15, 16, 17, 18, 19]])
  ['time', 'capitals']
  >>> time_caps.axis.capitals['berlin'::-1]
  DataArray([[ 2,  1,  0],
	 [ 7,  6,  5],
	 [12, 11, 10],
	 [17, 16, 15]])
  ['time', 'capitals']
  >>> time_caps.axis.time['0015':'1815']
  DataArray([[ 0,  1,  2,  3,  4],
	 [ 5,  6,  7,  8,  9],
	 [10, 11, 12, 13, 14]])
  ['time', 'capitals']
  >>> time_caps[:, 'london':3]
  DataArray([[ 1,  2],
	 [ 6,  7],
	 [11, 12],
	 [16, 17]])
  ['time', 'capitals']


The .start and .stop attributes of the slice object can be either None, an integer index, or a valid tick. They may even be mixed. The .step attribute, however, must be None or an nonzero integer.

**Note: currently integer ticks clobber indices.** For example::

  >>> centered_data = DataArray(np.random.randn(6), [ ('c_idx', range(-3,3)) ])
  >>> centered_data.axis.c_idx.make_slice( slice(0, 6, None) )
  (slice(3, 6, None),)

make_slice() first tries to look up the key parameters as ticks, and then sees if the key parameters can be used as simple indices. Thus 0 is found as index 3, and 6 is passed through as index 6.

Possible resolution 1
*********************

"larry" would make this distinction::

  >>> centered_data.axis.c_idx[ [0]:[2] ]
  >>> < returns underlying array from [3:5] >
  >>> centered_data.axis.c_idx[ 0:2 ]
  >>> < returns underlying array from [0:2] >

And I believe mixing of ticks and is valid also.

Possible resolution 2
*********************

Do not allow integer ticks -- cast to float perhaps

**Note**: this will be the solution. When validating ticks on an Axis, ensure that none of them ``isinstance(t, int)``


Possible resolution 3
*********************

Restrict access to tick based slicing to another special slicing object.

.. _broadcasting:

Broadcasting
------------

What about broadcasting between two named arrays, where the broadcasting
adds an axis? The broadcasted DataArray below, "a", takes on dummy dimensions that are taken to be compatible with the larger DataArray::

  >>> b = DataArray(np.ones((3,3)), labels=('x','y'))
  >>> a = DataArray(np.ones((3,)), labels=('y',))
  >>> res = 2*b - a
  >>> res
  DataArray([[ 0.,  0.,  0.],
	 [ 0.,  0.,  0.],
	 [ 0.,  0.,  0.]])

When there are unlabeled dimensions, they also must be consistently oriented across arrays when broadcasting::

  >>> b = DataArray(np.random.randn(3,2,4), ['x', None, 'y'])
  >>> a = DataArray(np.random.randn(2,4), [None, 'y'])
  >>> res = a + b
  >>> res
  DataArray([[[-0.19010426, -0.55643254, -1.89616528, -1.60534666],
	  [-1.34319297, -2.0147686 , -1.43270408,  0.27277437]],

	 [[-0.82144488,  2.12268969, -1.23886644, -1.85773148],
	  [ 0.11721121, -1.09646755, -1.02949198,  1.06404044]],

	 [[-0.3381559 , -0.43403438, -1.82946762, -1.12704282],
	  [ 1.22197036, -1.73950015, -2.23539961, -0.46131822]]])

We already know that if the dimension labels don't match, this won't be allowed (even though the shapes are correct)::

  >>> a = DataArray(np.ones((3,)), labels=('x',))
  >>> res = 2*b - a
  ------------------------------------------------------------
  Traceback (most recent call last):
  ...
  NamedAxisError: Axis labels are incompatible for a binary operation: ['x', 'y'], ['x']

But a numpy idiom for padding dimensions helps us in this case::

  >>> res = 2*b - a[:,None]
  >>> res
  DataArray([[ 1.,  1.,  1.],
	 [ 1.,  1.,  1.],
	 [ 1.,  1.,  1.]])

In other words, this scenario is also a legal combination::

  >>> a2 = a[:,None]
  >>> a2.labels
  ['x', None]
  >>> b + a2
  DataArray([[ 2.,  2.,  2.],
	 [ 2.,  2.,  2.],
	 [ 2.,  2.,  2.]])

The rule for dimension compatibility is that any two axes match if one of the following is true

* their (label, length) pairs are equal
* for one labeled Axis, the other (label, length) pair is equal to (None, 1)

**Question** -- what about this situation::

  >>> b = DataArray(np.ones((3,3)), labels=('x','y'))
  >>> a = DataArray(np.ones((3,1)), labels=('x','y'))
  >>> a+b
  DataArray([[ 2.,  2.,  2.],
	 [ 2.,  2.,  2.],
	 [ 2.,  2.,  2.]])

The broadcasting rules currently allow this combination. I'm inclined to allow it. Even though the axes are different lengths in ``a`` and ``b``, and therefore *might* be considered different logical axes, there is no actual information collision from ``a.axis.y``.

.. _iteration:

Iteration
---------

seems to work::

  >>> for foo in time_caps:
  ...     print foo
  ...     print foo.axes
  ... 
  [0 1 2 3 4]
  ['capitals']
  [Axis(label='capitals', index=0, ticks=['washington', 'london', 'berlin', 'paris', 'moscow'])]
  [5 6 7 8 9]
  ['capitals']
  [Axis(label='capitals', index=0, ticks=['washington', 'london', 'berlin', 'paris', 'moscow'])]
  [10 11 12 13 14]
  ['capitals']
  [Axis(label='capitals', index=0, ticks=['washington', 'london', 'berlin', 'paris', 'moscow'])]
  [15 16 17 18 19]
  ['capitals']
  [Axis(label='capitals', index=0, ticks=['washington', 'london', 'berlin', 'paris', 'moscow'])]
  >>> for foo in time_caps.T:
      print foo
      print foo.axes
  ... 
  [ 0  5 10 15]
  ['time']
  [Axis(label='time', index=0, ticks=['0015', '0615', '1215', '1815'])]
  [ 1  6 11 16]
  ['time']
  [Axis(label='time', index=0, ticks=['0015', '0615', '1215', '1815'])]
  [ 2  7 12 17]
  ['time']
  [Axis(label='time', index=0, ticks=['0015', '0615', '1215', '1815'])]
  [ 3  8 13 18]
  ['time']
  [Axis(label='time', index=0, ticks=['0015', '0615', '1215', '1815'])]
  [ 4  9 14 19]
  ['time']
  [Axis(label='time', index=0, ticks=['0015', '0615', '1215', '1815'])]


.. _transposition:

Transposition of Axes
---------------------

Transposition of a DataArray preserves the dimension labels, and updates the corresponding indices::

  >>> b.shape
  (3, 2, 4)
  >>> b.axes
  [Axis(label='x', index=0, ticks=None), Axis(label=None, index=1, ticks=None), Axis(label='y', index=2, ticks=None)]
  >>> b.T.shape
  (4, 2, 3)
  >>> b.T.axes
  [Axis(label='y', index=0, ticks=None), Axis(label=None, index=1, ticks=None), Axis(label='x', index=2, ticks=None)]

.. _label_updates:

Changing Labels on DataArrays
-----------------------------

Tricky Attributes
`````````````````

* .labels -- currently a mutable list of Axis.name attributes
* .axes -- currently a mutable list of Axis objects
* .axis -- a key-to-attribute dictionary

Need an event-ful way to change an Axis's label, such that all the above attributes are updated.

**Proposed solution**: 

1. use a set_label() method. This will consequently update the parent array's 
    (labels, axes, axis) attributes. 
2. make the mutable lists into *tuples* to deny write access.
3. make the KeyStruct ``.axis`` have write-once access 

ToDo
----

* Implementing axes with values in them (a la Per Sederberg)
* Support DataArray instances with mixed axes: simple ones with no values 
   and 'fancy' ones with data in them.  Syntax?

``DataArray.from_names(data, labels=['a','b','c'])``

``DataArray(data, axes=[('a',[1,2,3]), ('b',['one','two']), ('c',['red','black'])])``

``DataArray(data, axes=[('a',[1,2,3]), ('b',None), ('c',['red','black'])])``

* We need to support unnamed axes.
* Units support (Darren's)
* Jagged arrays? Kilian's suggestion.  Drop the base array altogether, and
access data via the .axis objects alone.
* "Enum dtype", could be useful for event selection.
* "Ordered factors"? Something R supports.
* How many axis classes?


Axis api: if a is an axis from an array: a = x.axis.a

a.at(key): return the slice at that key, with one less dimension than x
a.keep(keys): join slices for given keys, dims=dims(x)
a.drop(keys): like keep, but the opposite

a[i] valid cases:
i: integer => normal numpy scalar indexing, one less dim than x
i: slice: numpy view slicing.  same dims as x, must recover the ticks 
i: list/array: numpy fancy indexing, as long as the index list is 1d only.


