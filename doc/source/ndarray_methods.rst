.. testsetup::

    import numpy as np

=======
Methods
=======

Here is a list of the ``array`` methods:

.. we got the method names with

    >>> a = np.random.randn(3,4)
    >>> filter(lambda x: type(getattr(a,x))==type(a.min), dir(a))

* '__array__',
* :ref:`'__array_prepare__',<explicitly_redef>`
* :ref:`'__array_wrap__',<explicitly_redef>`
* '__copy__',
* '__deepcopy__',
* :ref:`'__new__',<explicitly_redef>`
* '__reduce__',
* '__reduce_ex__',
* '__setstate__',
* 'all',
* 'any',
* :ref:`'argmax', <wrapped_reduction_special>`
* :ref:`'argmin', <wrapped_reduction_special>`
* :ref:`'argsort',<sorting_methods>`
* 'astype',
* 'byteswap',
* :ref:`'choose',<wtf_methods>`
* 'clip',
* 'compress',
* 'conj',
* 'conjugate',
* 'copy',
* :ref:`'cumprod',<incomplete_reductions>`
* :ref:`'cumsum',<incomplete_reductions>`
* :ref:`'diagonal',<wtf_methods>`
* 'dump',
* 'dumps',
* 'fill',
* :ref:`'flatten',<reshaping_methods>`
* 'getfield',
* 'item',
* 'itemset',
* :ref:`'max',<wrapped_reduction>`
* :ref:`'mean',<wrapped_reduction>`
* :ref:`'min',<wrapped_reduction>`
* 'newbyteorder',
* 'nonzero',
* :ref:`'prod',<wrapped_reduction>`
* :ref:`'ptp',<wrapped_reduction>`
* 'put',
* :ref:`'ravel',<reshaping_methods>`
* :ref:`'repeat',<incomplete_reductions>`
* :ref:`'reshape',<reshaping_methods>`
* :ref:`'resize',<reshaping_methods>`
* 'round',
* :ref:`'searchsorted',<wtf_methods>`
* 'setfield',
* 'setflags',
* :ref:`'sort',<sorting_methods>`
* :ref:`'squeeze',<reshaping_methods>`
* :ref:`'std',<wrapped_reduction>`
* :ref:`'sum',<wrapped_reduction>`
* :ref:`'swapaxes',<explicitly_redef>`
* :ref:`'take',<incomplete_reductions>`
* 'tofile',
* 'tolist',
* 'tostring',
* 'trace',
* :ref:`'transpose',<explicitly_redef>`
* :ref:`'var',<wrapped_reduction>`
* 'view']

.. _sorting_methods:

Sorting
-------

sort() and argsort()

These methods default to sorting the flattened array (returning an
ndarray). If given an axis keyword, then it is possible to preserve
the axes meta-data *only if* there are no ticks on the sorted
Axis. Otherwise, an ndarray is returned.

.. _explicitly_redef:

Explicitly overloaded
---------------------

These methods do not fit into a simple pattern, and are explicitly overloaded
in the DataArray class definition.

.. _wrapped_reduction:

Regular reductions (eg, min)
----------------------------

These methods are wrapped in a generic runner that pays attention to which axis
is being trimmed out (if only one), and then sets the remaining axes on the
resulting array. It also allows for the translation of Axis-name to Axis-index.

.. _wrapped_reduction_special:

Special reductions (eg, argmin)
-------------------------------

These methods are currently wrapped as a generic reduction. 

These methods return an index, or an array of indices into the array in
question. That significantly changes the model of the array in question. Should
the return type here NOT be DataArray?

.. _incomplete_reductions:

Accumulations
-------------

These methods are wrapped in a generic accumulator.

These methods have the property of taking an "axis" keyword argument, and yet
not eliminating that axis. They also default to working on the flattened array
if the axis parameter is left unspecified.

.. _wtf_methods:

Not-applicable methods
----------------------

Possibly N/A methods?

.. _reshaping_methods:

Reshapes
--------

Reshaping is prickly.. I've already implemented certain slicing
mechanisms that can insert unlabeled axes with length-1. This seems
legitimate. Also squeezing out length-1 seems legitimate (**even if
the Axis is labeled?**). 

The reshaping currently only trims or pads the array shape with 1s, or
flattens the array entirely (returning an ndarray).

