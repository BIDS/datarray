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

For me, this looks like the domain of utility functions.

