import distutils.version as v

import numpy as np

if v.LooseVersion(np.__version__) < v.LooseVersion('1.4'):
    raise ImportError('Numpy version > 1.4 is required to use datarray')

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print "No datarray unit testing available."
    
from version import __version__
from datarray import DataArray
