"""
Functions for pretty-printing tabular data, such as a DataArray, as a grid.
"""
import numpy as np
import itertools

class GridDataFormatter(object):
    """
    Defines reasonable defaults for a data formatter.
    """
    def __init__(self, data=None):
        self.data = data

    def min_width(self):
        return 1
    
    def standard_width(self):
        return 9

    def max_width(self):
        if self.data is None: return 75
        return max([len(unicode(val)) for val in self.data.flat])

    def format(self, value, width=None):
        """
        Formats a given value to a fixed width.
        """
        if width is None: width = self.standard_width()
        return '{0:<{width}}'.format(value, width=width)[:width]
    
    def format_all(self, values, width=None):
        """
        Formats an array of values to a fixed width, returning a string array.
        """
        if width is None: width = self.standard_width()
        out = np.array([self.format(value, width) for value in values.flat])
        return out.reshape(values.shape)

class FloatFormatter(GridDataFormatter):
    def __init__(self, data, sign=False, strip_zeros=True):
        GridDataFormatter.__init__(self, data)
        flat = data.flatten()
        absolute = np.abs(flat.compress((flat != 0) & ~np.isnan(flat) & ~np.isinf(flat)))
        if sign: self.sign = '+'
        else: self.sign = ' '
        self.strip_zeros = strip_zeros
        if len(absolute):
            self.max_val = np.max(absolute)
            self.min_val = np.min(absolute)
            self.leading_digits = max(1, int(np.log10(self.max_val)) + 1)
            self.leading_zeros = max(0, int(np.ceil(-np.log10(self.min_val))))
        else:
            self.max_val = self.min_val = 0
            self.leading_digits = 1
            self.leading_zeros = 0
        self.large_exponent = (self.leading_digits >= 101) or (self.leading_zeros >= 100)

    def min_width(self):
        return min(self._min_width_standard(), self._min_width_exponential())

    def _min_width_standard(self):
        # 1 character for sign
        # enough room for all the leading digits
        # 1 character for decimal point
        # enough room for all the leading zeros
        # 1 more digit
        return self.leading_digits + self.leading_zeros + 3

    def _min_width_exponential(self):
        # enough room for -3.1e+nn or -3.1e+nnn
        return self.large_exponent + 8

    def standard_width(self):
        return self.min_width() + 2

    def max_width(self):
        return self.leading_digits + 8

    def format(self, value, width=None):
        if width is None: width = self.standard_width()
        if width < self._min_width_standard():
            return self._format_exponential(value, width)
        else:
            return self._format_standard(value, width)

    def _format_exponential(self, value, width):
        precision = max(1, width - 7 - self.large_exponent)
        return '{0:<{sign}{width}.{precision}e}'.format(value,
                                                        width=width,
                                                        sign=self.sign,
                                                        precision=precision)

    def _format_standard(self, value, width):
        precision = max(1, width - 2 - self.leading_digits)
        result = '{0:>{sign}{width}.{precision}f}'.format(value, width=width,
                                                          sign=self.sign,
                                                          precision=precision)
        if self.strip_zeros:
            return '{0:<{width}}'.format(result.rstrip('0'), width=width)
        else: return result
    
    def format_all(self, values, width=None):
        """
        Formats an array of values to a fixed width, returning a string array.
        """
        if width is None: width = self.standard_width()
        if width < self._min_width_standard():
            formatter = self._format_exponential
        else:
            formatter = self._format_standard

        out = np.array([formatter(value, width) for value in values.flat])
        return out.reshape(values.shape)

class IntFormatter(FloatFormatter):
    """
    The IntFormatter tries to just print all the digits of the ints, but falls
    back on being a FloatFormatter if there isn't room.
    """
    def _min_width_standard(self):
        return self.leading_digits + 1
    
    def standard_width(self):
        return self._min_width_standard()

    def _format_standard(self, value, width):
        return '{0:>{sign}{width}d}'.format(value, width=width, sign=self.sign)

class BoolFormatter(GridDataFormatter):
    def standard_width(self):
        return 5

    def max_width(self):
        return 5

    def format(self, value, width=5):
        if width < 5:
            if value: return 'T'
            else: return '-'
        else:
            if value: return ' True'
            else: return 'False'

class StrFormatter(GridDataFormatter):
    def min_width(self):
        return min(3, self.max_width())

    def standard_width(self):
        return min(9, self.max_width())

class ComplexFormatter(GridDataFormatter):
    def __init__(self, data):
        GridDataFormatter.__init__(self, data)
        self.real_format = FloatFormatter(data, strip_zeros=False)
        self.imag_format = FloatFormatter(data, strip_zeros=False, 
                                          sign=True)

    def min_width(self):
        return max(self.real_format.min_width(),
                   self.imag_format.min_width())*2 + 1

    def standard_width(self):
        return max(self.real_format.standard_width(),
                   self.imag_format.standard_width())*2 + 1

    def max_width(self):
        return max(self.real_format.max_width(),
                   self.imag_format.max_width())*2
    
    def format(self, value, width=None):
        #TODO: optimize
        if width is None: width = self.standard_width()
        part_width = (width-1)//2
        real_part = self.real_format.format(value.real, part_width)
        imag_part = self.imag_format.format(value.imag, part_width)
        result = '{0}{1}j'.format(real_part, imag_part)
        return '{0:<{width}}'.format(result, width=width)

def get_formatter(arr):
    typeobj = arr.dtype.type
    if issubclass(typeobj, np.bool): return BoolFormatter(arr)
    elif issubclass(typeobj, np.int): return IntFormatter(arr)
    elif issubclass(typeobj, np.floating): return FloatFormatter(arr)
    elif issubclass(typeobj, np.complex): return ComplexFormatter(arr)
    else: return StrFormatter

def grid_layout(arr):
    formatter = get_formatter(arr)
    layout = formatter.format_all(arr[:6, :6], 9)
    return layout

def layout_to_string(str_array, cell_width, row_header=None, col_header=None):
    assert str_array.ndim == 2
    assert str_array.shape[0] > 0
    assert str_array.shape[1] > 0
    
    width = cell_width * str_array.shape[1] - 1
    height = str_array.shape[0]
    chars = np.zeros((height, width), dtype='|S1')
    chars.fill(' ')

    for r in xrange(str_array.shape[0]):
        for c in xrange(str_array.shape[1]):
            entry = str_array[r, c]
            cstart = c*cell_width
            cend = min(cstart + len(entry), width)
            chars[r, cstart:cend] = list(entry)[:cend-cstart]
    return '\n'.join([''.join(row) for row in chars])

