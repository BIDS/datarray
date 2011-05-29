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
    
    # get an "axes indexer" object for the indicated objects.
    >>> a.axes('stocks', 'date')

This indexer object returns something that is meant to be indexed with as many
dimensions as it was passed arguments, but that will, upon indexing, return
arrays with dimensions ordered just like the original underlying array.
    
The reason that I think that this is more natural is that the information that
you have is all available at the point where you are constructing the slicer,
you don't need to go rummaging around the code to find the correct order of the
axes from where the array was originally defined.  It also potentially permits
you to use underlying arrays with different axis orders in the same code
unambiguously.

There was also the thought that with numerical arguments that this would fill a
hole in the current numpy API for arbitrary re-ordering of axes in a view for
slicing (essentially a super-generalized transpose-ish sort of thing)

I think that the result of the slicing operation retains the original ordering,
but the slices provided to a.axes()[] need to match the order of the arguments
to a.axes.

So in other words, when you do


tslicer = a.axes('t')

then

tslicer['a':'z']

returns an array with axes x, y, z, t in that order, but sliced as
a[:,:,:,'a':'z'] 

When you have:

xyslicer = a.axes('x', 'y')
yxslicer = a.axes('y', 'x')

then I would expect to do:

xyslicer[x1:x2, y1:y2]

but

yxslicer[y1:y2, x1:x2]

However, these are two equivalent ways of writing a[x1:x2, y1:y2, :, :]



::
      
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
    
    # do mapped indexing XXX - this might not have been the final decision
    >>> a.axes('stocks', 'date', mapped=True)['aapl':'goog', datetime.date(2011, 1, 1):datetime.date(2011, 5, 14)]

    # For mapped indexing
    
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
