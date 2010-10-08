======
Issues
======

Questions and issues about the datarray prototype.

.. contents::


Ticks
=====

Ticks are a relatively new addition to datarrays. The labels of a datarrays
identify the axes of the array. The ticks of a datarray identify the elements
along an axis. Both labels and ticks are optional.          

Axis._tick_dict is not updated when ticks are changed
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Example::

    >> dar = DataArray([1, 2], [('time', ['A', 'B'])])
    >> dar.axis.time._tick_dict 
       {'A': 0, 'B': 1}
    >> dar.axis.time.ticks[0] = 'X'
    >> dar.axis.time.ticks
       ['X', 'B']
    >> dar.axis.time._tick_dict 
       {'A': 0, 'B': 1}

Possible solutions:

#. Don't allow ticks to be changed
#. Only allow ticks to be changed through a method that also updates _tick_dict
#. Don't store _tick_dict, create on the fly as needed

pandas, I believe, makes the ticks immutable (#1). larry allows the ticks to
be changed and calculates the mapping dict on the fly (#3).   


Can I have ticks without labels?
""""""""""""""""""""""""""""""""

I'd like to use ticks without labels. At the moment that is not possible::

    >>> DataArray([1, 2], [(None, ('a', 'b'))])
    <snip>
    ValueError: ticks only supported when Axis has a label
    
Well, it is possible::

    >>> dar = DataArray([1, 2], [('tmp', ('a', 'b'))])
    >>> dar.set_label(0, None)
    >>> dar.axes
    (Axis(label=None, index=0, ticks=('a', 'b')),)    


Add a ticks input parameter?
""""""""""""""""""""""""""""

What do you think of adding a ``ticks`` parameter to DataArray?

Current behavior::

    >>> dar = DataArray([[1, 2], [3, 4]], (('row', ['A','B']), ('col', ['C', 'D'])))
    >>> dar.axes
    (Axis(label='row', index=0, ticks=['A', 'B']),
     Axis(label='col', index=1, ticks=['C', 'D']))

Proposed ticks as separate input parameter::

    >>> DataArray([[1, 2], [3, 4]], labels=('row', 'col'), ticks=[['A', 'B'], ['C', 'D']])

I think this would make it easier for new users to construct a DataArray with
ticks just from looking at the DataArray signature. It would match the
signature of Axis. My use case is to use ticks only and not names axes (at
first), so::

    >>> DataArray([[1, 2], [3, 4]], ticks=[['A', 'B'], ['C', 'D']])

instead of the current::

    >>> DataArray([[1, 2], [3, 4]], ((None, ['A','B']), (None, ['C', 'D'])))

It might also cause less typos (parentheses matching) at the command line.

Having separate labels and ticks input parameters would also leave the option
open to allow any hashable object, like a tuple, to be used as a label.
Currently tuples have a special meaning, the (labels, ticks) tuple.

Create Axis._tick_dict when needed?
"""""""""""""""""""""""""""""""""""

How about creating Axis._tick_dict on the fly when needed (but not saving it)?

**Pros**

- Faster datarray creation (it does look like you get _tick_dict for free
  since you need to check that the ticks are unique anyway, but set()
  is faster)
- Faster datarray copy
- Use less memory
- Easier to archive
- Simplify Axis
- Prevent user from doing ``dar.axes[0]._tick_dict['a'] = 10``
- Catches (on calls to ``make_slice`` and ``keep``) user mischief like
  dar.axes[0].ticks = ('a', 'a')
- No need to update Axis._tick_dict when user changes ticks  

**Cons**

- Slower ``make_slice``
- Slower ``keep``


Axis, axes
==========

Datarrays were created from the need to label the axes of a numpy array.

datarray1 + datarrat2 = which axes?
"""""""""""""""""""""""""""""""""""

Which axes are returned by binary operations?

Make two datarrays::

    >> dar1 = DataArray([1, 2], [('time', ['A1', 'B1'])])
    >> dar2 = DataArray([1, 2], [('time', ['A2', 'B2'])])

``dar1`` on the left-hand side::
 
    >> dar12 = dar1 + dar2
    >> dar12.axes
       (Axis(label='time', index=0, ticks=['A1', 'B1']),)

``dar1`` on the right-hand side::
 
    >> dar21 = dar2 + dar1
    >> dar21.axes
       (Axis(label='time', index=0, ticks=['A2', 'B2']),)

So a binary operation returns the axes from the left-hand side? No. Seems the
left most non-None axes are used::

    >> dar3 = DataArray([1, 2])
    >> dar31 = dar3 + dar1
    >> dar31.axes
       (Axis(label='time', index=0, ticks=['A1', 'B1']),)

So binary operation may returns parts of both axes::

    >> dar1 = DataArray([[1, 2], [3, 4]], [None, ('col', ['A', 'B'])])
    >> dar2 = DataArray([[1, 2], [3, 4]], [('row', ['a', 'b']), None])
    >> dar12 = dar1 + dar2
    >> dar12.axes
       
    (Axis(label='row', index=0, ticks=['a', 'b']),
     Axis(label='col', index=1, ticks=['A', 'B']))
     
Is that the intended behavior?            

Why does Axis.__eq__ require the index to be equal?
"""""""""""""""""""""""""""""""""""""""""""""""""""

Example::

    >> dar1 = DataArray([[1, 2], [3, 4]], [('row', ['r0', 'r1']), ('col', ['c0', 'c1'])])
    >> dar2 = DataArray([[1, 2], [3, 4]], [('col', ['c0', 'c1']), ('row', ['r0', 'r1'])])
    >> dar1.axes[0] == dar2.axes[1]
       False
             
Axis, axis, axes
""""""""""""""""

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

could be replaces with
::
    if isinstance(key, tuple):
        self.axes = self.axes[key]
        
So it would pull out the axes logic from DataArray and place it in Axes.

Should DataArray.axes be a list instead of a tuple?
"""""""""""""""""""""""""""""""""""""""""""""""""""

Why not make DataArrya.axes a list instead of a tuple? Then user can replace
an axis from one datarray to another, can pop an Axis, etc.   


Can axis labels be anything besides None or str?
""""""""""""""""""""""""""""""""""""""""""""""""

from http://projects.scipy.org/numpy/wiki/NdarrayWithNamedAxes: "Axis labels
(the name of a dimension) must be valid Python identifiers." I don't know
what that means.

It would be nice if axis labels could be anything hashable like str,
datetime.date(), int, tuple.

But labels must be strings to do indexing like this::

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
""""""""""""""""""""""""""""""""""""""""""" 

Set up data::

    >> import numpy as np
    >> N = 100
    >> arr = np.random.rand(N, N)
    >> idx1 = map(str, range(N))
    >> idx2 = map(str, range(N))

Time the creation of a datarray::

    >> from datarray import DataArray
    >> import datarray
    >> labels = [('row', idx1), ('col', idx2)]
    >> timeit datarray.DataArray(arr, labels)
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
    >> label = [idx1, idx2]
    >> timeit la.larry(arr, label)
    100000 loops, best of 3: 13.5 us per loop
    >> timeit la.larry(arr, label, integrity=False)
    1000000 loops, best of 3: 1.25 us per loop

Also both datarray and DataMatrix make a mapping dictionary when the data
object is created---that takes time. larry makes a mapping dictionary on the
fly, when needed.

Why is the time to create a datarray important? Because even an operation as
simple as ``dar1 + dar2`` creates a datarray.

Direct access to array?
"""""""""""""""""""""""

Labels and ticks add overhead. Sometimes, after aligning my datarrays, I would
like to work directly with the numpy arrays. Is there a way to do that with
datarrays?

For example, with a labeled array, `larry <http://github.com/kwgoodman/la>`_,
the underlying numpy array is always accessable as the attribute ``x``::

    >>> import la
    >>> lar = la.larry([1, 2, 3])
    >>> lar.x
    array([1, 2, 3])
    >>> lar.x = myfunc(lar.x)
    
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
    
But base is not gauranteed to be a view. What's another solution? Could create
an attribute at init time, but that slows down init.    


Alignment
=========

Datarray may not handle alignment directly. But some users of datarrays would
like an easy way to align datarrays.
     
Support for alignment?
""""""""""""""""""""""

Will datarray provide any support for those who want binary operations between
two datarrays to join labels or ticks using various join methods?

`A use case <http://larry.sourceforge.net/work.html#alignment>`_ from
`larry <http://larry.sourceforge.net>`_:

By default, binary operations between two larrys use an inner join of the
labels (the intersection of the labels)::

    >>> lar1 = larry([1, 2])
    >>> lar2 = larry([1, 2, 3])
    >>> lar1 + lar2
    label_0
        0
        1
    x
    array([2, 4])

The sum of two larrys using an outer join (union of the labels)::

    >>> la.add(lar1, lar2, join='outer')
    label_0
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
"""""""""""""""""""""""""""""

What's an outer join (or inner, left, right) along an axis of two datarrays if
one datarray has ticks and the other doesn't?

Background:

It is often useful to align two datarrays before performing binary operations
such as +, -, *, /. Two datarrays are aligned when both datarrays have the same
labels and ticks along all axes.

Aligned::

    >> dar1 = DataArray([1, 2])
    >> dar2 = DataArray([3, 4])
    >> dar1.axes == dar2.axes
       True

Unaligned::

    >> dar1 = DataArray([1, 2], labels=("time",))
    >> dar2 = DataArray([3, 4], labels=("distance",))
    >> dar1.axes == dar2.axes
       False

Unaligned but returns aligned since Axis.__eq__ doesn't (yet) check for
equality of ticks::

    >> dar1 = DataArray([1, 2], labels=[("time", ['A', 'B'])])
    >> dar2 = DataArray([1, 2], labels=[("time", ['A', 'different'])])
    >> dar1.axes == dar2.axes
       True

Let's say we make an add function with user control of the join method::

    >>> add(dar1, dar2, join='outer')

Since datarray allows empty axis labels (None) and ticks (None), what does an
outer join mean if dar1 has ticks but dar2 doesn't::

    >>> dar1 = DataArray([1, 2], labels=[("time", ['A', 'B'])])
    >>> dar2 = DataArray([1, 2], labels=[("time",)])
    
What would the following return?
::
    >>> add(dar1, dar2, join='outer')
    
larry requires all axes to have ticks, if none are given then the ticks default
to range(n).

datarray.reshape
""""""""""""""""

Reshape operations scramble labels and ticks. Some numpy functions and
array methods use reshape. Should reshape convert a datarray to an array?

Looks like datarray will need unit tests for every numpy function and array
method.


Misc
==== 

Miscellaneous observation on datarrays.     

How do I save a datarray in HDF5 using h5py?
""""""""""""""""""""""""""""""""""""""""""""

`h5py <http://h5py.alfven.org>`_, which stores data in HDF5 format, can only
save numpy arrays.

What are the parts of a datarray that need to be saved? And can they be stored
as numpy arrays?

A datarray can be broken down to the following components:

- data (store directly as numpy array)
- labels (store as object array since it contains None and str and covert
  back on load?)
- ticks (each axis stored as numpy array with axis number stored as HDF5
  Dataset attribute, but then ticks along any one axis must be homogenous
  in dtype)
- Dictionary of tick index mappings (ignore, recreate on load)
    
(I need to write a function that saves an Axis object to HDF5.)

If I don't save Axis._tick_dict, would I have to worry about a user changing
the mapping?
::
    >>> dar.axes[0]
    Axis(label='one', index=0, ticks=('a', 'b'))
    >>> dar.axes[0]._tick_dict
    {'a': 0, 'b': 1}
    >>> dar.axes[0]._tick_dict['a'] = 10
    >>> dar.axes[0]._tick_dict
    {'a': 10, 'b': 1}
   

Can labels and ticks be changed?
""""""""""""""""""""""""""""""""  

Ticks can be changed::

    >>> dar = DataArray([1, 2], [('row', ['A','B'])])
    >>> dar.axes
    (Axis(label='row', index=0, ticks=['A', 'B']),)
    >>> dar.axes[0].ticks[0] = 'CHANGED'
    >>> dar.axes
    (Axis(label='row', index=0, ticks=['CHANGED', 'B']),)
    
But Axis._tick_dict is not updated when user changes ticks.    

And so can labels::

    >>> dar.set_label(0, 'new label')
    >>> dar   
    DataArray([1, 2])
    ('new label',)

