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
