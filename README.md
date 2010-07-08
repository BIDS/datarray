# *datarray:* Numpy arrays with named axes #

## Introduction

Scientists, engineers, mathematicians and statisticians don't just work with matrices; Nay, they often work with structured data, just like you'd find in a table. However, functionality for this is missing from Numpy, and there are efforts to create something to fill the void.

This is one of those efforts. Currently, this is very experimental. The API *will* change, make no mistake.

## Prior Art

At present, there is no accepted standard solution to dealing with tabular data such as this. However, based on the following list of ad-hoc and proposal-level implementations of something such as this, there is *definitely* a demand for it For examples, in no particular order:

* [Tabular](http://bitbucket.org/elaine/tabular/src) implements a spreadsheet-inspired datatype, with rows/columns, csv/etc. IO, and fancy tabular operations.

* [scikits.statsmodels](http://scikits.appspot.com/statsmodels) sounded as though it had some features we'd like to eventually see implemented on top of something such as datarray, and [Skipper](http://scipystats.blogspot.com/) seemed pretty interested in something like this himself.

* [scikits.timeseries](http://scikits.appspot.com/timeseries) also has a time-series-specific object that's somewhat reminiscent of labeled arrays.

* [pandas](http://pandas.sourceforge.net/) is based around a number of DataFrame-esque datatypes.

* [pydataframe](http://code.google.com/p/pydataframe/) is supposed to be a clone of R's data.frame.

* [larry](http://github.com/kwgoodman/la), or "labeled array," often comes up in discussions alongside pandas.

* [divisi](http://github.com/commonsense/divisi2) includes labeled sparse and dense arrays.

* Finally, of course, there is [Fernando Perez's original work on datarray](http://www.github.com/fperez/datarray), of which this is a fork.

## Project Goals

1. Get something akin to this in the numpy core.

2. Stick to basic functionality such that projects like scikits.statsmodels and pandas can use it as a base datatype.

3. Make an interface that allows for simple, pretty manipulation that doesn't introduce confusion.

4. Oh, and make sure that the base numpy array is still accessible.

## TODO:

1. Nail down the interface.
2. Agree on what should and shouldn't be in datarray
