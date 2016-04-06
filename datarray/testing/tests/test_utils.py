"Tests of datarray unit test utilities"

import numpy as np
from numpy.testing import assert_raises

from datarray.datarray import DataArray
from datarray.testing.utils import assert_datarray_equal

def test_assert_datarray_equal():
    # Test assert_datarray_equal
    
    x = DataArray([1, 2])
    y = DataArray([1, 2])
    assert_datarray_equal(x, y, "Should not raise assertion")
    y = DataArray([1, 3])
    assert_raises(AssertionError, assert_datarray_equal, x, y)
    y = DataArray([1, 2, 3])
    assert_raises(AssertionError, assert_datarray_equal, x, y)
    y = DataArray([1, 2], 'a')
    assert_raises(AssertionError, assert_datarray_equal, x, y)    
    y = DataArray([1, 2], [('a', ['a', 'b'])])
    assert_raises(AssertionError, assert_datarray_equal, x, y)    
    
    x = DataArray([1, 2], 'a')
    y = DataArray([1, 2], 'a')
    assert_datarray_equal(x, y, "Should not raise assertion")
    y = DataArray([1, 2], 'b')       
    assert_raises(AssertionError, assert_datarray_equal, x, y)
    y = DataArray([1, 2], [('b', ['a', 'b'])])       
    assert_raises(AssertionError, assert_datarray_equal, x, y)
        
    x = DataArray([1, 2], 'a')    
    y = DataArray([1, 2], [('a', None)])       
    assert_datarray_equal(x, y, "Should not raise assertion")
    
    x = DataArray([[1, 2], [3, 4]], [('ax1', ['a', 'b']), ('ax2', ['a', 'b'])])
    y = DataArray([[1, 2], [3, 4]], [('ax1', ['a', 'b']), ('ax2', ['a', 'b'])])
    assert_datarray_equal(x, y, "Should not raise assertion")
    y = DataArray([[1, 2], [3, 4]], [('ax1', ['X', 'b']), ('ax2', ['a', 'b'])])
    assert_raises(AssertionError, assert_datarray_equal, x, y)
    y = DataArray([[1, 2], [3, 4]], [('ax1', ['a', 'b']), ('ax2', None)])    
    assert_raises(AssertionError, assert_datarray_equal, x, y)
    y = DataArray([[9, 2], [3, 4]], [('ax1', ['a', 'b']), ('ax2', ['a', 'b'])])        
    assert_raises(AssertionError, assert_datarray_equal, x, y)    
    
    x = DataArray([1, np.nan])
    y = DataArray([1, np.nan])
    assert_datarray_equal(x, y, "Should not raise assertion")
    
    x = DataArray([1, 2], 'a')
    y = 1      
    assert_raises(AssertionError, assert_datarray_equal, x, y)
    y = np.array([1, 2])
    assert_raises(AssertionError, assert_datarray_equal, x, y)          
    
    x = 1
    y = 2
    assert_raises(AssertionError, assert_datarray_equal, x, y)
    x = np.array([1])
    y = np.array([2])
    assert_raises(AssertionError, assert_datarray_equal, x, y)        
