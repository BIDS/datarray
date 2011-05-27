"""Arrays with rich geometric semantics.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
# Stdlib
import distutils.version as v

# Third-party
import numpy as np
if v.LooseVersion(np.__version__) < v.LooseVersion('1.4'):
    raise ImportError('Numpy version > 1.4 is required to use datarray')

# Our own
try:
    from .testing.testlib import test
except ImportError:
    print "No datarray unit testing available."
    
from .version import __version__
from .datarray import DataArray
