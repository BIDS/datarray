========
 Pandas
========

Overview
^^^^^^^^

Pandas provides a timeseries and stack-of-timeseries objects. They
seem heavily geared towards financial data. Despite the fact of
**being** an ndarray, Pandas objects seem to be a specialized
alternative to ndarrays rather than an augmentation of them.

Features

* **Is** an ndarray 
* axes are not named
* Is dict-like, with respect to its indices (ticks)
* If ticks are indices, semantics of indexing are ambiguous
* Separate objects from 1D and 2D, no support for n>2


Indexing
********

Point-indexing syntax can use ticks or integer indices. Range indexing
only works with integers, but uses the same syntax

Semantic Ambiguity
------------------

Integer tick values interfere with integer indexing, for example::

    >>> t = pandas.Series.fromValue(1.0, range(5,0,-1), 'i')
    >>> t[:] = np.random.randint(100, size=5)
    >>> t
    5    23
    4    62
    3    66
    2    91
    1    91
    >>> t[2] = 0
    >>> t
    5    23
    4    62
    3    66
    2    0
    1    91



Binary Operations
*****************

Alignment
---------

If data is partially aligned, missing data is filled with NaNs. This
introduces a union with respect to the "range" of the data. This also
will **cast** the data to floating point.::

   >>> t.dtype
    dtype('int32')
   >>> t - t[:3]
   5    0.0
   4    0.0
   3    0.0
   2    NaN
   1    NaN
