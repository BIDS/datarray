=========
DataArray
=========

Arrays with named / labeled axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


ndarrays extended to have an explicit "hypercross" of axes, each with names (possibly defaulted). 

* for methods in which an "axis" is denoted, an axis name may be used

* for all manners under which dimension numbers and lengths must match, also the hypercrosses must be consistent

* broadcasting will "inherit" labels from the super-hyper-cross

* padding dimensions will insert "dummy" dimensions, eg: ::

   a = datarray( np.random.randn(10,10), ('time', 'temp') )
   a[:,None,:].axes --> ('time', None, 'temp') 

* axes may be transposed

Arrays with named axes, whose named axes have ticks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

each named axis has tick labels

* numpy, fancy and slice-like indexing on each axis: ::

   x.named_axis[...]
   --> does any kind of numpy indexing on the axis
   x.named_axis.at( *args )
   --> returns essentially "fancy" indexing along the axis, at valid ticks in args
   x.named_axis.t_slice( start, stop, [step])
   --> where arguments are valid ticks, performs a slicing-like operation along the axis

* mixed indexing on the array: ::

   x.at( *args )
   --> len(args) <= x.ndim -- for each indexing spec in args, perform that indexing
          on the enumerated axes
   x.t_slice( *args )
   --> same as above, but perform t_slice slicing on the enumerated axes


Lessons Learned
^^^^^^^^^^^^^^^

"Smart" Indexing
****************

The smart indexing implemented by Larry is very full featured. I believe the design of using lists to separating labels from integers in mixed indexing is a good choice (and necessary). However, I think it illustrates the potential confusion created by mixed indexing and is a good argument for discouraging/not allowing it.

"Smart" Arithmetic
******************

* Larry makes attempts to align its arrays when performing arithmetic, so as to operate on identical coordinates. 
* It also might introduce intersections between arrays. 
* It does not broadcast


