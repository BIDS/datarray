======================================
 Issues, open questions and todo list
======================================

Questions and issues about the datarray prototype.

.. contents::


Labels
======

Labels are a relatively new addition to datarrays. The labels of a datarrays
identify the axes of the array. The labels of a datarray identify the elements
along an axis. Both labels and labels are optional.

Axis._label_dict is not updated when labels are changed
-------------------------------------------------------

Example::

    >> dar = DataArray([1, 2], [('time', ['A', 'B'])])
    >> dar.axis.time._label_dict
       {'A': 0, 'B': 1}
    >> dar.axis.time.labels[0] = 'X'
    >> dar.axis.time.labels
       ['X', 'B']
    >> dar.axis.time._label_dict
       {'A': 0, 'B': 1}

Possible solutions:

#. Don't allow labels to be changed
#. Only allow labels to be changed through a method that also updates _label_dict
#. Don't store _label_dict, create on the fly as needed

pandas, I believe, makes the labels immutable (#1). larry allows the labels to
be changed and calculates the mapping dict on the fly (#3).


Can I have labels without axis names?
-------------------------------------

I'd like to use labels without names. At the moment that is not possible::

    >>> DataArray([1, 2], [(None, ('a', 'b'))])
    <snip>
    ValueError: labels only supported when Axis has a name

Well, it is possible::

    >>> dar = DataArray([1, 2], [('tmp', ('a', 'b'))])
    >>> dar.set_name(0, None)
    >>> dar.axes
    (Axis(name=None, index=0, labels=('a', 'b')),)


Add a labels input parameter?
-----------------------------

What do you think of adding a ``labels`` parameter to DataArray?

Current behavior::

    >>> dar = DataArray([[1, 2], [3, 4]], (('row', ['A','B']), ('col', ['C', 'D'])))
    >>> dar.axes
    (Axis(name='row', index=0, labels=['A', 'B']),
     Axis(name='col', index=1, labels=['C', 'D']))

Proposed labels as separate input parameter::

    >>> DataArray([[1, 2], [3, 4]], names=('row', 'col'), labels=[['A', 'B'], ['C', 'D']])

I think this would make it easier for new users to construct a DataArray with
labels just from looking at the DataArray signature. It would match the
signature of Axis. My use case is to use labels only and not names axes (at
first), so::

    >>> DataArray([[1, 2], [3, 4]], labels=[['A', 'B'], ['C', 'D']])

instead of the current::

    >>> DataArray([[1, 2], [3, 4]], ((None, ['A','B']), (None, ['C', 'D'])))

It might also cause less typos (parentheses matching) at the command line.

Having separate names and labels input parameters would also leave the option
open to allow any hashable object, like a tuple, to be used as a name.
Currently tuples have a special meaning, the (names, labels) tuple.

Create Axis._label_dict when needed?
------------------------------------

How about creating Axis._label_dict on the fly when needed (but not saving it)?

**Pros**

- Faster datarray creation (it does look like you get _label_dict for free
  since you need to check that the labels are unique anyway, but set()
  is faster)
- Faster datarray copy
- Use less memory
- Easier to archive
- Simplify Axis
- Prevent user from doing ``dar.axes[0]._label_dict['a'] = 10``
- Catches (on calls to ``make_slice`` and ``keep``) user mischief like
  dar.axes[0].labels = ('a', 'a')
- No need to update Axis._label_dict when user changes labels

**Cons**

- Slower ``make_slice``
- Slower ``keep``


Axis, axes
==========

Datarrays were created from the need to name the axes of a numpy array.

datarray1 + datarrat2 = which axes?
-----------------------------------

Which axes are returned by binary operations?

Make two datarrays::

    >> dar1 = DataArray([1, 2], [('time', ['A1', 'B1'])])
    >> dar2 = DataArray([1, 2], [('time', ['A2', 'B2'])])

``dar1`` on the left-hand side::

    >> dar12 = dar1 + dar2
    >> dar12.axes
       (Axis(name='time', index=0, labels=['A1', 'B1']),)

``dar1`` on the right-hand side::

    >> dar21 = dar2 + dar1
    >> dar21.axes
       (Axis(name='time', index=0, labels=['A2', 'B2']),)

So a binary operation returns the axes from the left-hand side? No. Seems the
left most non-None axes are used::

    >> dar3 = DataArray([1, 2])
    >> dar31 = dar3 + dar1
    >> dar31.axes
       (Axis(name='time', index=0, labels=['A1', 'B1']),)

So binary operation may returns parts of both axes::

    >> dar1 = DataArray([[1, 2], [3, 4]], [None, ('col', ['A', 'B'])])
    >> dar2 = DataArray([[1, 2], [3, 4]], [('row', ['a', 'b']), None])
    >> dar12 = dar1 + dar2
    >> dar12.axes

    (Axis(name='row', index=0, labels=['a', 'b']),
     Axis(name='col', index=1, labels=['A', 'B']))

Is that the intended behavior?

Why does Axis.__eq__ require the index to be equal?
---------------------------------------------------

Example::

    >> dar1 = DataArray([[1, 2], [3, 4]], [('row', ['r0', 'r1']), ('col', ['c0', 'c1'])])
    >> dar2 = DataArray([[1, 2], [3, 4]], [('col', ['c0', 'c1']), ('row', ['r0', 'r1'])])
    >> dar1.axes[0] == dar2.axes[1]
       False

Axis, axis, axes
----------------

The functions, classes, and methods that take care of axes are:

- Axis (class)
- DataArray.axis (meth)
- DataArray.axes (meth)
- _reordered_axes (func)
- _expand_ellipsis (func)
- _make_singleton_axes (func)

I find having both DataArray.axis and DataArray.axes confusing at first. I
wonder if it would simplify things if there was only:

- Axes (class)
- Data.axes (instance of Axes)

That would consolidate everything in the Axes class. For example, in
DataArray.__getitem__ this::

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

could be replaced with::

    if isinstance(key, tuple):
        self.axes = self.axes[key]

So it would pull out the axes logic from DataArray and place it in Axes.

Should DataArray.axes be a list instead of a tuple?
---------------------------------------------------

Why not make DataArray.axes a list instead of a tuple? Then user can replace
an axis from one datarray to another, can pop an Axis, etc.


Can axis names be anything besides None or str?
-----------------------------------------------

from http://projects.scipy.org/numpy/wiki/NdarrayWithNamedAxes: "Axis names
(the name of a dimension) must be valid Python identifiers." I don't know
what that means.

It would be nice if axis names could be anything hashable like str,
datetime.date(), int, tuple.

But names must be strings to do indexing like this::

    >>> dar = DataArray([[1, 2], [3, 4]], (('row', ['A','B']), ('col', ['C', 'D'])))
    >>> dar.axis.row['A']
    DataArray([1, 2])
    ('col',)

One way to make it work would be to rewrite the above as::

    >>> dar.axis['row']['A']
    DataArray([1, 2])
    ('col',)

which would also make it easier to loop through the axes by name::

    >>> for axisname in ['row', col']:
   ....:    dar.axis[axisname][idx]
   ....:    ...


Performance
===========

Performance is not the primary concern during the prototype phase of datarray.
But some attention to performance issue will help guide the development of
datarrays.

How long does it take to create a datarray?
-------------------------------------------

Set up data::

    >> import numpy as np
    >> N = 100
    >> arr = np.random.rand(N, N)
    >> idx1 = map(str, range(N))
    >> idx2 = map(str, range(N))

Time the creation of a datarray::

    >> from datarray import DataArray
    >> import datarray
    >> names = [('row', idx1), ('col', idx2)]
    >> timeit datarray.DataArray(arr, names)
    1000 loops, best of 3: 160 us per loop

Time the creation of a pandas DataMatrix. A DataMatrix it is also a subclass
of numpy's ndarray, but it has been optimized so should be a proxy for how
fast a datarray can become::

    >> import pandas
    >> timeit pandas.DataMatrix(arr, idx1, idx2)
    10000 loops, best of 3: 50.7 us per loop

larry is not a subclass of numpy's ndarray, I think that is one reason it is
faster to create::

    >> import la
    >> name = [idx1, idx2]
    >> timeit la.larry(arr, name)
    100000 loops, best of 3: 13.5 us per loop
    >> timeit la.larry(arr, name, integrity=False)
    1000000 loops, best of 3: 1.25 us per loop

Also both datarray and DataMatrix make a mapping dictionary when the data
object is created---that takes time. larry makes a mapping dictionary on the
fly, when needed.

Why is the time to create a datarray important? Because even an operation as
simple as ``dar1 + dar2`` creates a datarray.

Direct access to array?
-----------------------

Names and labels add overhead. Sometimes, after aligning my datarrays, I would
like to work directly with the numpy arrays. Is there a way to do that with
datarrays?

For example, with a named array, larry_, the underlying numpy array is always
accessible as the attribute ``x``::

    >>> import la
    >>> lar = la.larry([1, 2, 3])
    >>> lar.x
    array([1, 2, 3])
    >>> lar.x = myfunc(lar.x)

.. _larry: http://github.com/kwgoodman/la
    
This might be one solution (base)::

    >> from datarray import DataArray
    >> x = DataArray([[1,2],[3,4]], [('row', ['r1', 'r2']), ('col', ['c1', 'c2'])])
    >> timeit x + x
    10000 loops, best of 3: 61.4 us per loop
    >> timeit x.base + x.base
    100000 loops, best of 3: 2.16 us per loop

and::

    >> x = DataArray([1, 2])
    >> x.base[0] = 9
    >> x

    DataArray([9, 2])
    (None,)

But base is not guaranteed to be a view. What's another solution? Could create
an attribute at init time, but that slows down init.


Alignment
=========

Datarray may not handle alignment directly. But some users of datarrays would
like an easy way to align datarrays.

Support for alignment?
----------------------

Will datarray provide any support for those who want binary operations between
two datarrays to join names or labels using various join methods?

`A use case <http://larry.sourceforge.net/work.html#alignment>`_ from larry_:

By default, binary operations between two larrys use an inner join of the
names (the intersection of the names)::

    >>> lar1 = larry([1, 2])
    >>> lar2 = larry([1, 2, 3])
    >>> lar1 + lar2
    name_0
        0
        1
    x
    array([2, 4])

The sum of two larrys using an outer join (union of the names)::

    >>> la.add(lar1, lar2, join='outer')
    name_0
        0
        1
        2
    x
    array([  2.,   4.,  NaN])

The available join methods are inner, outer, left, right, and list. If the
join method is specified as a list then the first element in the list is the
join method for axis=0, the second element is the join method for axis=1, and
so on.

How can datarrays be aligned?
-----------------------------

What's an outer join (or inner, left, right) along an axis of two datarrays if
one datarray has labels and the other doesn't?

Background:

It is often useful to align two datarrays before performing binary operations
such as +, -, \*, /. Two datarrays are aligned when both datarrays have the same
names and labels along all axes.

Aligned::

    >> dar1 = DataArray([1, 2])
    >> dar2 = DataArray([3, 4])
    >> dar1.axes == dar2.axes
       True

Unaligned::

    >> dar1 = DataArray([1, 2], names=("time",))
    >> dar2 = DataArray([3, 4], names=("distance",))
    >> dar1.axes == dar2.axes
       False

Unaligned but returns aligned since Axis.__eq__ doesn't (yet) check for
equality of labels::

    >> dar1 = DataArray([1, 2], names=[("time", ['A', 'B'])])
    >> dar2 = DataArray([1, 2], names=[("time", ['A', 'different'])])
    >> dar1.axes == dar2.axes
       True

Let's say we make an add function with user control of the join method::

    >>> add(dar1, dar2, join='outer')

Since datarray allows empty axis names (None) and labels (None), what does an
outer join mean if dar1 has labels but dar2 doesn't::

    >>> dar1 = DataArray([1, 2], names=[("time", ['A', 'B'])])
    >>> dar2 = DataArray([1, 2], names=[("time",)])

What would the following return?
::

    >>> add(dar1, dar2, join='outer')

larry requires all axes to have labels, if none are given then the labels default
to range(n).

datarray.reshape
----------------

Reshape operations scramble names and labels. Some numpy functions and
array methods use reshape. Should reshape convert a datarray to an array?

Looks like datarray will need unit tests for every numpy function and array
method.


Misc
====

Miscellaneous observation on datarrays.

How do I save a datarray in HDF5 using h5py?
--------------------------------------------

`h5py <http://h5py.alfven.org>`_, which stores data in HDF5 format, can only
save numpy arrays.

What are the parts of a datarray that need to be saved? And can they be stored
as numpy arrays?

A datarray can be broken down to the following components:

- data (store directly as numpy array)
- names (store as object array since it contains None and str and covert
  back on load?)
- labels (each axis stored as numpy array with axis number stored as HDF5
  Dataset attribute, but then labels along any one axis must be homogeneous
  in dtype)
- Dictionary of label index mappings (ignore, recreate on load)

(I need to write a function that saves an Axis object to HDF5.)

If I don't save Axis._label_dict, would I have to worry about a user changing
the mapping?
::

    >>> dar.axes[0]
    Axis(name='one', index=0, labels=('a', 'b'))
    >>> dar.axes[0]._label_dict
    {'a': 0, 'b': 1}
    >>> dar.axes[0]._label_dict['a'] = 10
    >>> dar.axes[0]._label_dict
    {'a': 10, 'b': 1}


Can names and labels be changed?
--------------------------------

Labels can be changed::

    >>> dar = DataArray([1, 2], [('row', ['A','B'])])
    >>> dar.axes
    (Axis(name='row', index=0, labels=['A', 'B']),)
    >>> dar.axes[0].labels[0] = 'CHANGED'
    >>> dar.axes
    (Axis(name='row', index=0, labels=['CHANGED', 'B']),)

But Axis._label_dict is not updated when user changes labels.

And so can names::

    >>> dar.set_name(0, 'new name')
    >>> dar
    DataArray([1, 2])
    ('new name',)

Fancy Indexing
--------------

It's not implemented at all yet.

.. _name_updates:

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
