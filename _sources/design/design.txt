==============================
 DataArray: some design notes
==============================

A DataArray is a subclass of the basic Numpy ndarray object that provides an
explicit mechanism for attaching information to the *axes* of the underlying
numpy array.  This is achieved by attaching an Axis object to each dimension of
the array; an Axis object has an optional *name* as well as optional *labels*
(think of them as tick labels in a figure).

With Axis objects attached to an array, it becomes possible to manipulate the
array by named axis, to slice an axis by named label, etc.  These features
complement the rich semantics that numpy has for the *contents* of an array,
encapsulated its dtype machinery for structured/record arrays.

Arrays with named / labeled axes
================================

ndarrays extended to have an explicit "hypercross" of axes, each with
names (possibly defaulted). 

* for methods in which an "axis" is denoted, an axis name may be used

* indexing/slicing along a named axis returns that slicing, at that axis,
  along with slice(None) slicing along all other axes    

* for all arithmetic/binary-op matters under which dimension numbers and
  lengths must match, also the hypercrosses must be consistent

* broadcasting will "inherit" labels from the super-hyper-cross
  (see np.broadcast)

* padding dimensions will insert "dummy" dimensions, eg::

   a = datarray( np.random.randn(10,10), ('time', 'temp') )
   a[:,None,:].axes --> ('time', None, 'temp') 

* axes may be transposed

Arrays with named axes, whose named axes have ticks
===================================================

each named axis has tick labels

* numpy, fancy and slice-like indexing on each axis::

   x.named_axis[...]
   --> does any kind of numpy indexing on the axis
   x.named_axis.at( *args )
   --> returns essentially "fancy" indexing along the axis, at valid ticks in args
   x.named_axis.t_slice( start, stop, [step])
   --> where arguments are valid ticks, performs a slicing-like operation along the axis

* mixed indexing on the array::

   x.at( *args )
   --> len(args) <= x.ndim -- for each indexing spec in args, perform that indexing
          on the enumerated axes
   x.t_slice( *args )
   --> same as above, but perform t_slice slicing on the enumerated axes

(my thoughts on) What Is The DataArray?
=======================================

* 1st and foremost, **an ndarray**, in N dimensions, with any dtype
* has means to locate data more descriptively (IE, with custom names
  for dimensions/axes, and custom names for indices along any axis)

::

  >>> darr = DataArray(np.random.randn(2,3,4), ('ex', 'why', 'zee'))
  >>> darr.sum(axis='ex')
  DataArray([[-0.39052695, -2.07493873,  1.19664474,  0.36681094],
	 [-1.04287781,  0.5767191 , -0.35425298,  1.10468356],
	 [ 0.08331866, -0.36532857,  0.12905265, -1.94559672]])
  ('why', 'zee')
  >>> for subarr in darr.axis.why:
  ...     print subarr.shape, subarr.labels
  ... 
  (2, 4) ('ex', 'zee')
  (2, 4) ('ex', 'zee')
  (2, 4) ('ex', 'zee')

* An axis "label" can always stand in for an axis number; an index
  "tick" can (in some TBD sense) stand in for an integer index
* if anything is **more restrictive** in operations, for example

::

  >>> ndarr_ones = np.ones((10,10,10))
  >>> ndarr_twos = np.ones((10,10,10))*2
  >>> ndarr_3s = ndarr_ones + ndarr_twos # OK!
  >>> darr_abc = DataArray(ndarr_ones, ('a', 'b', 'c'))
  >>> darr_bac = DataArray(ndarr_twos, ('b', 'a', 'c'))
  >>> darr_wtf = darr_abc + darr_bac # BAD! frames are rotated

(and my very own thoughts on) What The DataArray Is Not
=======================================================

Unions And Intersections
------------------------

DataArray may broadcast with certain union rules for adapting
metadata, but it does not do any data union/intersection rule for
operations. For example, the result of adding an array with axes ('a', 'c') with an
array with axis 'c' takes on information from the "superset" of
axes. This is analogous to ndarray taking on shape information from
the superset of shapes.

::

  >>> darr_abc[:,0,:]
  DataArray([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
	 ...
	 [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])
  ('a', 'c')
  >>> darr_bac[0,0]
  DataArray([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
  ('c',)
  >>> darr_abc[:,0,:] + darr_bac[0,0]
  DataArray([[ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
	 ...
	 [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.]])
  ('a', 'c')

But it will not fill or trim any dimension to fit the shape of a
fellow operand's array (it seems this violation is simply caught at the C-level of an ndarray)::

  >>> darr_abc[:,0,:] + darr_bac[0,0,:5]
  ------------------------------------------------------------
  Traceback (most recent call last):
    File "<ipython console>", line 1, in <module>
  ValueError: shape mismatch: objects cannot be broadcast to a single shape

For me, this looks like the **domain of utility functions** (or
possibly utility methods that yield new DataArrays).

Namespace
---------

It would be good practice to keep all the dynamically generated
DataArray attributes (eg, Axis labels) removed from the top-level
array attribute list. This is what we currently have as "axis". 

It might(?) be a good idea to put all future special purpose methods
under that object too.

   
Lessons Learned
===============

"Smart" Indexing
----------------

The smart indexing implemented by Larry is very full featured. I believe the
design of using lists to separating labels from integers in mixed indexing is a
good choice (and necessary). However, I think it illustrates the potential
confusion created by mixed indexing and is a good argument for discouraging/not
allowing it.

"Smart" Arithmetic
------------------

* Larry makes attempts to align its arrays when performing arithmetic, so as to
  operate on identical coordinates.
* It also might introduce intersections between arrays. 
* It does not broadcast

Ideas
=====

Axis Slicing
------------

Use Case: chained axis slicing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

slicing on an axis returns a new DataArray::

  arr = DataArray(np.random.randn(10,10), labels=('time', 'freq'))
  arr.axis.time[:5] --> new DataArray with (time, freq) axes

However, slicing on the special slicing object "aix" returns a new Special
Tuple (stuple). 

Stuple:

* is len-N, for ND arrays
* only one entry is (potentially) not ``slice(None)``
* has knowledge of its own index
* has knowledge of other axes (static or dynamically generated attributes)
* can be composed with other stuples in a special way (??) --

::

  s1 --> ( slice(0,4), slice(None) )
  s2 --> ( slice(None), slice(3,10) )
  s1 <compose> s2 --> ( slice(0,4), slice(3,10) )

* can be given a "parent" stuple when constructed, into which the new stuple
  merges its own slicing in ``__getitem__``

Constructor prototype::

  def __init__(self, *args, parent=None, index=None, name=None) ??

To chain slicing, the syntax would be like this::

  arr.aix.time[:4].freq[3:8]
  --OR--
  arr[ arr.aix.time[:4].freq[3:8] ]

Chaining an axis on itself **will not** be implemented yet (possibly ever)::

  arr.aix.time[:4].time[:2] --> raise error


============================================
 The May 2011 DataArray summit at Enthought
============================================

How to handle datarray indexing
===============================

This document is a summary of the syntax and semantics that was agreed upon at
the Data Array summit held at Enthought in May 2011.

The DataArray object will have a .axes attribute which exhibits the following
behaviour::

    >>> a = DataArray( ..., axes=('date', ('stocks', ('aapl', 'ibm' 'goog', 'msft')), 'metric'))
    
    # get the axis object
    >>> a.axes.stocks
    
    # the same as a[:,0:2,:]
    >>> a.axes.stocks['aapl':'goog']
    
    # get the nth axis object (particularly if not named)
    >>> a.axes[n]
    
    # get an "axes indexer" object for the indicated objects.
    >>> a.axes('stocks', 'date')

This indexer object returns something that is meant to be indexed with as many
dimensions as it was passed arguments, but that will, upon indexing, return
arrays with dimensions ordered just like the original underlying array.
    
The information that is all available at the point where you are constructing
the slicer, so you don't need to go rummaging around the code to find the
correct order of the axes from where the array was originally defined.  It also
potentially permits you to use underlying arrays with different axis orders in
the same code unambiguously.

There was also the thought that with numerical arguments that this would fill a
hole in the current numpy API for arbitrary re-ordering of axes in a view for
slicing (essentially a super-generalized transpose-ish sort of thing)

The result of the slicing operation retains the original ordering, but the
slices provided to a.axes()[] need to match the order of the arguments to
a.axes. So in other words, when you do::

    >>> tslicer = a.axes('t')

then::

    >>> tslicer['a':'z']

returns an array with axes x, y, z, t in that order, but sliced as::

    a[:,:,:,'a':'z'] 

When you have::

    xyslicer = a.axes('x', 'y')
    yxslicer = a.axes('y', 'x')

then I would expect to do::

    xyslicer[x1:x2, y1:y2]

but::

    yxslicer[y1:y2, x1:x2]

However, these are two equivalent ways of writing ``a[x1:x2, y1:y2, :, :]``.
If explicit transposition of the returned data is desired, it can be done
with::

    >>> a.transpose('stocks','date').axes('stocks','date')[...]

    # Now, actually do the slicing: equivalent to a[100, 0:2, :]
    >>> a.axes('stocks', 'date')['aapl':'goog',100]
    
    # can supply an axis number as well
    >>> a.axes(1, 'date')['aapl':'goog',100:200]

In addition axes can have the notion of a index mapper which allows indexing and
slicing by labels or values other than strings and integers.  To use these, you
have to supply a keyword argument to the axes call::
    
    # add a datetime.date -> index map
    >>> date_mapper = DictMapper(...)
    >>> a = DataArray( ..., axes=(('date', date_mapper), ... ))
    
    # do mapped indexing XXX - this might not have been the final decision
    >>> a.axes('stocks', 'date', mapped=True)['aapl':'goog', datetime.date(2011, 1, 1):datetime.date(2011, 5, 14)]

    # For mapped indexing
    
The exact semantics of mapping are yet to be determined, but the thought is that
there would be standard mappers to do things like interpolation, mapped integer
indexing.

Other notes
-----------

* Axis names can only be strings that are valid Python identifiers.
* Labels can only be strings, and must be unique.
* All other indexing cases are handled by mapping (however that will work).
* Axes can have arbitrary aliases which do not have to be unique.
* An axis can have an associated array of the same length as the set of labels
  for additional data storage.
