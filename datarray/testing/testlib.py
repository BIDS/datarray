"""Module defining the main test entry point exposed at the top level.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib
import sys

# Third-party
import nose
import nose.plugins.builtin
from nose.core import TestProgram

#-----------------------------------------------------------------------------
# Functions and classes
#-----------------------------------------------------------------------------

def test(doctests=True, extra_argv=None, **kw):
    """Run the nitime test suite using nose.

    Parameters
    ----------
    doctests : bool, optional  (default True)
      If true, also run the doctests in all docstrings.

    kw : dict
      Any other keywords are passed directly to nose.TestProgram(), which
      itself is a subclass of unittest.TestProgram().
    """
    # We construct our own argv manually, so we must set argv[0] ourselves
    argv = [ 'nosetests',
             # Name the package to actually test, in this case nitime
             'datarray',
             
             # extra info in tracebacks
             '--detailed-errors',

             # We add --exe because of setuptools' imbecility (it blindly does
             # chmod +x on ALL files).  Nose does the right thing and it tries
             # to avoid executables, setuptools unfortunately forces our hand
             # here.  This has been discussed on the distutils list and the
             # setuptools devs refuse to fix this problem!
             '--exe',
             ]

    if doctests:
        argv.append('--with-doctest')

    if extra_argv is not None:
        argv.extend(extra_argv)

    # Now nose can run
    TestProgram(argv=argv, exit=False, **kw)


# Tell nose that the test() function itself isn't a test, otherwise we get a
# recursive loop inside nose.
test.__test__ = False
