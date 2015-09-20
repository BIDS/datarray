import numpy as np
from datarray.datarray import DataArray
from datarray.print_grid import datarray_to_string

def test_2d_datarray_to_string():
    grid_string = """
country   year                                             
--------- -------------------------------------------------
          1994      1998      2002      2006      2010     
Netherlan  0.        0.142857  0.285714  0.428571  0.571429
Uruguay    0.714286  0.857143  1.        1.142857  1.285714
Germany    1.428571  1.571429  1.714286  1.857143  2.      
Spain      2.142857  2.285714  2.428571  2.571429  2.714286
    """.strip()
    
    test_array = np.arange(20).reshape((4, 5)) / 7.0
    row_spec = 'country', ['Netherlands', 'Uruguay', 'Germany', 'Spain']
    col_spec = 'year', list(map(str, [1994, 1998, 2002, 2006, 2010]))

    d_arr = DataArray(test_array, [row_spec, col_spec])
    assert datarray_to_string(d_arr) == grid_string


def test_1d_datarray_to_string():
    grid_string = """
country                                
---------------------------------------
Netherla  Uruguay   Germany   Spain    
 0.        0.714286  1.428571  2.142857
    """.strip()
    
    test_array = np.arange(20).reshape((4, 5)) / 7.0
    row_spec = 'country', ['Netherlands', 'Uruguay', 'Germany', 'Spain']
    col_spec = 'year', list(map(str, [1994, 1998, 2002, 2006, 2010]))

    d_arr = DataArray(test_array, [row_spec, col_spec])
    assert datarray_to_string(d_arr.axes.year['1994']) == grid_string

