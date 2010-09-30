"""datarray version information"""

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 3
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

Datarray provides a subclass of Numpy ndarrays that support:

- individual dimensions (axes) being labeled with meaningful descriptions
- labeled 'ticks' along each axis
- indexing and slicing by named axis
- indexing on any axis with the tick labels instead of only integers
- reduction operations (like .sum, .mean, etc) support named axis arguments
  instead of only integer indices.

  
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


Note
====

this code is currently experimental!  It is meant to be a place for the
community to understand and develop the right semantics and have a prototype
implementation that will ultimately (hopefully) be folded back into Numpy
itself.
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
PACKAGES            = ["datarray", "datarray/tests"]
PACKAGE_DATA        = {'datarray': ['LICENSE']}
REQUIRES            = ["numpy"]
