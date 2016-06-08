"""datarray unit testing utilities"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Third-party
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal

# Our own
from datarray.datarray import DataArray

__all__ = ['assert_datarray_equal']

#-----------------------------------------------------------------------------
# Functions and classes
#-----------------------------------------------------------------------------

def assert_datarray_equal(x, y, err_msg='', verbose=True):
    """
    Raise an AssertionError if two datarrays are not equal.

    Given two datarrays, assert that the shapes are equal, axes are equal, and
    all elements of the datarrays are equal. Given two scalars assert equality.
    In contrast to the standard usage in numpy, NaNs are compared like numbers,
    no assertion is raised if both objects have NaNs in the same positions.

    The usual caution for verifying equality with floating point numbers is
    advised.

    Parameters
    ----------
    x : {datarray, scalar}
        If you are testing a datarray method, for example, then this is the
        datarray (or scalar) returned by the method.   
    y : {datarray, scalar}
        This datarray represents the expected result. If `x` is not equal to
        `y`, then an AssertionError is raised.
    err_msg : str
        If `x` is not equal to `y`, then the string `err_msg` will be added to
        the top of the AssertionError message.
    verbose : bool
        If True, the conflicting values are appended to the error message.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If actual and desired datarrays are not equal.

    Examples
    --------
    If the two datarrays are equal then None is returned:

    >>> from datarray.testing import assert_datarray_equal
    >>> from datarray.datarray import DataArray
    >>> x = DataArray([1, 2])
    >>> y = DataArray([1, 2])
    >>> assert_datarray_equal(x, y)

    If the two datarrays are not equal then an AssertionError is raised:

    >>> x = DataArray([1, 2], ('time',))
    >>> y = DataArray([1, 2], ('distance',))
    >>> assert_datarray_equal(x, y)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "datarray/testing/utils.py", line 133, in assert_datarray_equal
        raise AssertionError, err_msg
    AssertionError:
    <BLANKLINE>
        ----------
        AXIS NAMES
        ----------
    <BLANKLINE>
        Items are not equal:
            item=0
    <BLANKLINE>
         ACTUAL: 'time'
         DESIRED: 'distance'
    <BLANKLINE>
    """
    # Initialize
    fail = []        
            
    # Function to make section headings
    def heading(text):
        line = '-' * len(text)
        return '\n\n' + line + '\n' + text + '\n' + line + '\n'
    
    # The assert depends on the type of x and y
    if np.isscalar(x) and np.isscalar(y):
    
        # Both x and y are scalars        
        try:
            assert_equal(x, y)
        except AssertionError as err:
            fail.append(heading('SCALARS') + str(err))
            
    elif (type(x) is np.ndarray) and (type(y) is np.ndarray):
    
        # Both x and y are scalars       
        try:
            assert_array_equal(x, y)
        except AssertionError as err:
            fail.append(heading('ARRAYS') + str(err))            
                
    elif (type(x) == DataArray) + (type(y) == DataArray) == 1:
    
        # Only one of x and y are datarrays; test failed
        try: 
            assert_equal(type(x), type(y))
        except AssertionError as err:
            fail.append(heading('TYPE') + str(err))
                                                   
    else:
        
        # Both x and y are datarrays
    
        # shape
        try:         
            assert_equal(x.shape, y.shape)
        except AssertionError as err:
            fail.append(heading('SHAPE') + str(err))       

        # axis names
        try:         
            assert_equal(x.names, y.names)
        except AssertionError as err:
            fail.append(heading('AXIS NAMES') + str(err))
            
        # labels
        for ax in range(x.ndim):
            try:
                assert_equal(x.axes[ax].labels, y.axes[ax].labels)
            except AssertionError as err:
                fail.append(heading('LABELS ALONG AXIS = %d' % ax) + str(err))                         

        # axes
        for ax in range(x.ndim):
            try:         
                assert_(x.axes[ax], y.axes[ax])
            except AssertionError as err:
                fail.append(heading('AXIS OBJECT ALONG AXIS = %d' % ax) + str(err))
                fail.append('x: ' + str(x.axes[ax]))
                fail.append('y: ' + str(y.axes[ax]))
                
        # data
        try:         
            assert_array_equal(x.base, y.base)
        except AssertionError as err:
            fail.append(heading('ARRAY') + str(err))                
    
    # Did the test pass?    
    if len(fail) > 0:
        # No
        if verbose:
            err_msgs = ''.join(fail)
            err_msgs = err_msgs.replace('\n', '\n\t')
            if len(err_msg):
                err_msg = heading("TEST: " + err_msg) + err_msgs
            else:
                err_msg = err_msgs           
            raise AssertionError(err_msg)
        else:
            raise AssertionError                    
        
