.. image:: https://travis-ci.org/BIDS/datarray.svg?branch=master
    :target: https://travis-ci.org/BIDS/datarray

######################################
Datarray: Numpy arrays with named axes
######################################

Scientists, engineers, mathematicians and statisticians don't just work with
matrices; they often work with structured data, just like you'd find in a
table. However, functionality for this is missing from Numpy, and there are
efforts to create something to fill the void.  This is one of those efforts.

.. warning::

   This code is currently experimental, and its API *will* change!  It is meant
   to be a place for the community to understand and develop the right
   semantics and have a prototype implementation that will ultimately
   (hopefully) be folded back into Numpy.

Datarray provides a subclass of Numpy ndarrays that support:

- individual dimensions (axes) being labeled with meaningful descriptions
- labeled 'ticks' along each axis
- indexing and slicing by named axis
- indexing on any axis with the tick labels instead of only integers
- reduction operations (like .sum, .mean, etc) support named axis arguments
  instead of only integer indices.

*********
Prior Art
*********

In no particular order:

* `xarray <http://xarray.pydata.org/en/stable>`_ - very close in spirit to
  this package, xarray implements named ND array axes and tick labels.  It
  integrates with (and depends on) Pandas.  If you are doing production work,
  and don't mind the pandas dependency, please use xarray rather than this
  package.  Xarray used to be called "xray".

* `pandas <http://pandas.pydata.org>`_ is based around a number of
  DataFrame-esque datatypes.

* `Tabular <http://bitbucket.org/elaine/tabular/src>`_ implements a
  spreadsheet-inspired datatype, with rows/columns, csv/etc. IO, and fancy
  tabular operations.

* `scikits.statsmodels <http://scikits.appspot.com/statsmodels>`_ sounded as
  though it had some features we'd like to eventually see implemented on top of
  something such as datarray, and `Skipper <http://scipystats.blogspot.com>`_
  seemed pretty interested in something like this himself.

* `scikits.timeseries <http://scikits.appspot.com/timeseries>`_ also has a
  time-series-specific object that's somewhat reminiscent of labeled arrays.

* `pydataframe <https://pypi.python.org/pypi/pydataframe>`_ is supposed to be a
  clone of R's data.frame.

* `larry <http://github.com/kwgoodman/la>`_, or "labeled array," often comes up
  in discussions alongside pandas.

* `divisi <http://github.com/commonsense/divisi2>`_ includes labeled sparse and
  dense arrays.

* `pymvpa <https://github.com/PyMVPA/PyMVPA>`_ provides Dataset class
  encapsulating the data together with matching in length sets of attributes
  for the first two (samples and features) dimensions.  Dataset is not a
  subclass of numpy array to allow other data structures (e.g. sparse
  matrices).

* `ptsa <http://git.debian.org/?p=pkg-exppsy/ptsa.git>`_ subclasses
  ndarray to provide attributes per dimensions aiming to ease slicing/indexing
  given the values of the axis attributes

*************
Project Goals
*************

1. Get something akin to this in the numpy core;
2. Stick to basic functionality such that projects like scikits.statsmodels can
   use it as a base datatype;
3. Make an interface that allows for simple, pretty manipulation that doesn't
   introduce confusion;
4. Oh, and make sure that the base numpy array is still accessible.

****
Code
****

You can find our sources and single-click downloads:

* `Main repository`_ on Github;
* Documentation_ for the current release;
* Download the `current trunk`_ as a tar/zip file;
* Downloads of all `available releases`_.

The latest released version is always available from `pypi
<https://pypi.python.org/pypi/datarray>`_.

*******
Support
*******

Please put up issues on the `datarray issue tracker
<https://github.com/bids/datarray/issues>`_.

.. _main repository: http://github.com/bids/datarray
.. _Documentation: http://bids.github.com/datarray
.. _current trunk: http://github.com/bids/datarray/archives/master
.. _available releases: http://github.com/bids/datarray/releases
