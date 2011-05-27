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
    
    # get an "axes indexer" object for the indicated objects
    >>> a.axes('stocks', 'date')
    
    # actually do the slicing: equivalent to a[100, 0:2, :]
    >>> a.axes('stocks', 'date')['aapl':'goog',100]
    
    # can supply an axis number as well
    >>> a.axes(1, 'date')['aapl':'goog',100:200]

In addition axes can have the notion of a index mapper which allows indexing and
slicing by labels or values other than strings and integers.  To use these, you
have to supply a keyword argument to the axes call::
    
    # add a datetime.date -> index map
    >>> date_mapper = DictMapper(...)
    >>> a = DataArray( ..., axes=(('date', date_mapper), ... ))
    
    # do mapped indexing
    >>> a.axes('stocks', 'date', mapped=True)['aapl':'goog', datetime.date(2011, 1, 1):datetime.date(2011, 5, 14)]

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
