====================
 Larray (aka Larry)
====================

Overview
^^^^^^^^

Larray offers the notion of "ticks", but the axes themselves are not named. The
model seems to be something like *data with coordinates*

Importantly,

* Pure Python implementation
* Is **not** an ndarray

  * therefore, lots of redefined functionality
  * also lots of presumed intention of data (shuffling labels, group means, ...)
  * not lightweight

* Does **not** offer named axes
* **Only one (class of) dtype!!**
* Can do n-D
* Good mixed indexing


Construction
************

Larrays can be constructed from an array-like object and tick names for each
axis. Alternatively, Larrays can be constructed from a number of
data-with-coordinates representations.


Here's how to create a larry using **fromtuples** (note the cast to float, and
the filled-in NaN)::

    >>> data = [('a', 'a', 1), ('a', 'b', 2), ('b', 'a', 3)]
    >>> larry.fromtuples(data)
    label_0
	a
	b
    label_1
	a
	b
    x
    array([[  1.,   2.],
	   [  3.,  NaN]])

Here are examples of **fromdict** and **fromlist**::

    >>> data = {('a', 'c'): 1, ('a', 'd'): 2, ('b', 'c'): 3, ('b', 'd'): 4}
    >>> larry.fromdict(data)
    label_0
	a
	b
    label_1
	c
	d
    x
    array([[ 1.,  2.],
	   [ 3.,  4.]])

    >>> data = [[1, 2, 3, 4], [('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd')]]
    >>> larry.fromlist(data)
    label_0
	a
	b
    label_1
	c
	d
    x
    array([[ 1.,  2.],
	   [ 3.,  4.]])           

Indexing
********

Indexing using the bracket syntax arr[ <slicer> ] seems to return you exactly
what numpy would slice out of the underlying array. All slicing works, with the
exception of "fancy" indexing, and ellipsis indexing, and the use of
**np.newaxis**.

There is also a smart slicer riding along with the larrays that can slice with
label information. It seems to nicely blend labels and regular integer slicing.
To disambiguate possible integer labels and integer indexing, labels always
must be enclosed in a list::

    >>> arr = la.larry(np.arange(6).reshape(2,3), [ ['u', 'v'], [2,5,3], ])
    >>> arr
    label_0
	u
	v
    label_1
	2
	5
	3
    x
    array([[0, 1, 2],
	   [3, 4, 5]])
    >>> arr.lix[['u']]
    label_0
	2
	5
	3
    x
    array([0, 1, 2])
    >>> arr.lix[['u'],2:5]
    2
    >>> arr.lix[['u'],[2]:[5]]
    0
    >>> arr.lix[['u'],[2]:[3]]
    label_0
	2
	5
    x
    array([0, 1])


Binary Operations (arithmetic)
******************************

Binary operations are not, in general, numpy-thonic

Alignment
---------

Larray seems to want to only make binary operations on data with identical
coordinates. Furthermore, it will re-align the data if necessary. Therefore,
this example is ok::

    >>> y1 = larry([1, 2], [['a', 'z']])
    >>> y2 = larry([1, 2], [['z', 'a']])
    
What is ``y1 + y2``?
::

    >>> y1 + y2
    label_0
        a
        z
    x
    array([3, 3])

But this fails::

    >>> z1 = larry([1, 2], [['a', 'b']])
    >>> z2 = larry([3, 4], [['c', 'd']])

    >>> z1 + z2
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "la/la/deflarry.py", line 494, in __add__
        x, y, label = self.__align(other)
      File "la/la/deflarry.py", line 731, in __align
        raise IndexError, 'A dimension has no matching labels'
    IndexError: A dimension has no matching labels

Intersections and Broadcasting
------------------------------

Binary ops can introduce an implicit intersection operation, for example (this
would be illegal code in numpy)::

    >>> arr = la.larry(np.arange(6).reshape(2,3), [ ['u', 'v'], ['x','y','z']])
    >>> arr2 = la.larry(np.arange(9).reshape(3,3), [ ['u', 'v', 'w'], ['x', 'y', 'z']] )
    >>> arr2 + arr
    label_0
	u
	v
    label_1
	x
	y
	z
    x
    array([[ 0,  2,  4],
	   [ 6,  8, 10]])


According to the matched-coordinates operation rule, broadcasting does not happen::

    >>> arr3 = la.larry([4,5,6], [['x','y','z']])
    >>> arr3 + arr
    ------------------------------------------------------------
    Traceback (most recent call last):
      File "<ipython console>", line 1, in <module>
      File "/Users/mike/usr/lib/python2.5/site-packages/la/deflarry.py", line 583, in __add__
	x, y, label = self.__align(other)
      File "/Users/mike/usr/lib/python2.5/site-packages/la/deflarry.py", line 820, in __align
	raise IndexError, msg
    IndexError: Binary operation on two larrys with different dimension
