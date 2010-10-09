"""datarray version information"""

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 6
__version__ = "%s.%s.%s" % (_version_major, _version_minor, _version_micro)


CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = "NumPy arrays with named axes and named indices."

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = """
========================================
 Datarray: Numpy arrays with named axes
========================================

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

Prior Art
=========

At present, there is no accepted standard solution to dealing with tabular data
such as this. However, based on the following list of ad-hoc and proposal-level
implementations of something such as this, there is *definitely* a demand for
it.  For examples, in no particular order:

* [Tabular](http://bitbucket.org/elaine/tabular/src) implements a
  spreadsheet-inspired datatype, with rows/columns, csv/etc. IO, and fancy
  tabular operations.

* [scikits.statsmodels](http://scikits.appspot.com/statsmodels) sounded as
  though it had some features we'd like to eventually see implemented on top of
  something such as datarray, and [Skipper](http://scipystats.blogspot.com/)
  seemed pretty interested in something like this himself.

* [scikits.timeseries](http://scikits.appspot.com/timeseries) also has a
  time-series-specific object that's somewhat reminiscent of labeled arrays.

* [pandas](http://pandas.sourceforge.net/) is based around a number of
  DataFrame-esque datatypes.

* [pydataframe](http://code.google.com/p/pydataframe/) is supposed to be a
  clone of R's data.frame.

* [larry](http://github.com/kwgoodman/la), or "labeled array," often comes up
  in discussions alongside pandas.

* [divisi](http://github.com/commonsense/divisi2) includes labeled sparse and
  dense arrays.

Project Goals
=============

1. Get something akin to this in the numpy core.

2. Stick to basic functionality such that projects like scikits.statsmodels and
pandas can use it as a base datatype.

3. Make an interface that allows for simple, pretty manipulation that doesn't
introduce confusion.

4. Oh, and make sure that the base numpy array is still accessible.

  
Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github.
* Documentation_ for all releases and current development tree.
* Download as a tar/zip file the `current trunk`_.
* Downloads of all `available releases`_.

.. _main repository: http://github.com/fperez/datarray
.. _Documentation: http://fperez.github.com/datarray-doc
.. _current trunk: http://github.com/fperez/datarray/archives/master
.. _available releases: http://github.com/fperez/datarray/downloads
"""


NAME                = 'datarray'
MAINTAINER          = "Numpy Developers"
MAINTAINER_EMAIL    = "numpy-discussion@scipy.org"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://github.com/fperez/datarray"
DOWNLOAD_URL        = "http://github.com/fperez/datarray/archives/master"
LICENSE             = "Simplified BSD"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "Datarray developers"
AUTHOR_EMAIL        = "numpy-discussion@scipy.org"
PLATFORMS           = "OS Independent"
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
ISRELEASED          = False
VERSION             = __version__
PACKAGES            = ["datarray", "datarray/tests", "datarray/testing"]
PACKAGE_DATA        = {'datarray': ['LICENSE']}
REQUIRES            = ["numpy"]
