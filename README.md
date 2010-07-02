============================================
# *datarray:* Numpy arrays with named axes #
============================================

## Introduction

Scientists, engineers, mathematicians and statisticians don't just work with matrices; Nay, they often work with structured data, just like you'd find in a table. However, functionality for this is missing from Numpy, and there are efforts to create something to fill the void.

This is one of those efforts. Currently, this is very experimental. The API *will* change, make no mistake.

## Prior Art

At present, there is no accepted standard solution to dealing with tabular data such as this. However, based on the following list of ad-hoc and proposal-level implementations of something such as this, there is *definitely* a demand for it For examples, in no particular order:

* Elaine and some friends wrote the slick-sounding *tabular*: *http://bitbucket.org/elaine/tabular/src*

* [scikits.statsmodels](http://scikits.appspot.com/statsmodels) sounded as though it had some features I'd like to eventually see implemented on top of something such as datarray, and [Skipper](http://scipystats.blogspot.com/) seemed pretty interested in something like this himself.

* [pandas](http://pandas.sourceforge.net/), like scikits, also builds on this sort of functionality.

* [pydataframe](http://code.google.com/p/pydataframe/) is supposed to be a clone of R's data.frame, which scikits.statsmodels and pandas both seem to be missing.

* [larry](http://github.com/kwgoodman/la), or labeled array, often comes up in discussions.

* Finally, of course, there is [Fernando Perez's original work on datarray](http://www.github.com/fperez/datarray), of which this is a fork.

## Project Goals

1. Get something akin to this in the numpy core.

2. Stick to basic functionality such that projects like scikits.statsmodels and pandas can use it as a base datatype.

3. Make an interface that allows for simple, pretty manipulation that doesn't introduce confusion.

4. Oh, and make sure that the base numpy array is still accessible.

## TODO:

1. Try to integrate self more fully into the scipy community.

2. Familiarize myself with the internal implementation of datarray.

3. Might as well look at larry and tabular as well!

4. (Maybe) Figure out how R's data.frame works.
