=========
DataArray
=========

Arrays with named / labeled axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Lessons Learned
^^^^^^^^^^^^^^^

"Smart" Indexing
****************

The smart indexing implemented by Larry is very full featured. I believe the
design of using lists to separating labels from integers in mixed indexing is a
good choice (and necessary). However, I think it illustrates the potential
confusion created by mixed indexing and is a good argument for discouraging/not
allowing it.

"Smart" Arithmetic
******************

* Larry makes attempts to align its arrays when performing arithmetic, so as to
  operate on identical coordinates.
* It also might introduce intersections between arrays. 
* It does not broadcast

Ideas
^^^^^

Axis Slicing
************

Use Case: chained axis slicing
------------------------------

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
