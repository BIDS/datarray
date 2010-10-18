Printing Datarrays
==================

One of the most important ways to understand what's going on in a labeled
array is to be able to see a pretty text representation of it. In Divisi2 I
stole the __str__ method from PySparse to accomplish this, but NumPy arrays are
more varied than PySparse (where everything is two-dimensional and made of
floats).

Can we build on NumPy's str?
----------------------------

NumPy has provided somewhat-pretty text representations for a long time, but
the code in numpy.core.arrayprint is

- difficult to extend
- undocumented
- kind of spaghetti, frankly
- largely untouched for the last 13 years!

Its output can be aesthetically suboptimal in some cases. When printing large
arrays of floats, for example, it will wrap every line like this:

    [[  0.00000000e+00   1.00000000e-04   2.00000000e-04 ...,   4.70000000e-03
        4.80000000e-03   4.90000000e-03]
     [  5.00000000e-03   5.10000000e-03   5.20000000e-03 ...,   9.70000000e-03
        9.80000000e-03   9.90000000e-03]
     [  1.00000000e-02   1.01000000e-02   1.02000000e-02 ...,   1.47000000e-02
        1.48000000e-02   1.49000000e-02]
     ..., 
     [  3.35000000e-01   3.35100000e-01   3.35200000e-01 ...,   3.39700000e-01
        3.39800000e-01   3.39900000e-01]
     [  3.40000000e-01   3.40100000e-01   3.40200000e-01 ...,   3.44700000e-01
        3.44800000e-01   3.44900000e-01]
     [  3.45000000e-01   3.45100000e-01   3.45200000e-01 ...,   3.49700000e-01
        3.49800000e-01   3.49900000e-01]]

The user can understand what that means, but it'll be hard to stick labels on.

My conclusion is that it will be better to build this representation from the
ground up.

The 2D pretty-printer
---------------------
Screens are 2D, so everything is a variant of the 2D case. What we need is a
class designed for printing strings in a grid. This class will then:

- Find a formatter for the dtype of the matrix (the "cell formatter").
- Make an array (a string array? might as well) of equal-width string
  representations
- Attach row and column labels as the first row and column of the array
- Join together everything into a correctly-aligned, multi-line string

The width of each cell is a negotiation between the grid formatter and the cell
formatter:

- Cell: I can print these floats in 5 to 15 characters. More characters is
  better, of course.
- Grid: I'll give you 7.
- Cell: Stingy bastard.

Maybe this could be accomplished with "small", "medium", and "large" options
for each formatter, allowing us to reuse arrayprint formatters:

- float: large = high precision, medium = lower precision, small = lower
  precision and suppress_small
- int: large = max number of digits, medium/small = exponential notation
- str: large = maximum length, medium = truncate
- bool: large = ' True'/'False', medium/small = 'T'/'-' (to be visually
  distinct)

Brackets are _not_ printed (it's too hard to work them in with the labels).

The 1D pretty-printer
---------------------
It's the 2D printer with only one row.

The 3D pretty-printer
---------------------
When people work with n-dimensional labeled data and n>2, what they often do
is flatten it out into 2 dimensions. The rows are single data points, and the
columns are all the indices followed by the value. Show a few of these from the
beginning of the matrix, dots, and a few of these from the end of the matrix.

Then put all those back into the grid-maker.

If there are more than 30 or so dimensions, we are sad.

