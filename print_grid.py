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
        return min(self.leading_digits + 8, 16)

    def format(self, value, width=None):
        if width is None: width = self.standard_width()
        if self._use_exponential_format(width):
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
    
    def _use_exponential_format(self, width):
        return (width < self._min_width_standard())

    def format_all(self, values, width=None):
        """
        Formats an array of values to a fixed width, returning a string array.
        """
        if width is None: width = self.standard_width()
        if self._use_exponential_format(width):
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

def grid_layout(arr, width=75, height=10):
    # get the maximum possible amount we'd be able to display
    array_sample = arr[:height, :width//2]
    formatter = get_formatter(arr)
    
    # first choice: show the whole array at full width
    cell_width = formatter.max_width()
    columns_shown = arr.shape[1]
    column_ellipsis = False

    if (cell_width+1) * columns_shown > width+1:
        # second choice: show the whole array at at least standard width
        standard_width = formatter.standard_width()
        cell_width = (width+1) // (columns_shown) - 1
        if cell_width < standard_width:
            # third choice: show at least 5 columns at standard width
            column_ellipsis = True
            cell_width = standard_width
            columns_shown = (width-3) // (cell_width+1)
            if columns_shown < 5:
                # fourth choice: as many columns as possible at minimum width
                cell_width = formatter.min_width()
                columns_shown = max(1, (width-3) // (cell_width+1))
    cells_shown = arr[:height, :columns_shown]
    layout = formatter.format_all(cells_shown, cell_width)
    
    ungrid = [list(row) for row in layout]
    
    if column_ellipsis:
        ungrid[0].append('...')

    if height < arr.shape[0]: # row ellipsis
        ungrid.append(['...'])
    
    return ungrid, cells_shown

def labeled_layout(arr, width=75, height=10, row_label_width=9):
    inner_width, inner_height = width, height
    if arr.axis[0].ticks:
        inner_width = width - row_label_width-1
    if arr.axis[1].ticks:
        inner_height -= 1
    row_header = (arr.axis[0].ticks and arr.axis[0].name)
    col_header = (arr.axis[1].ticks and arr.axis[1].name)
    if row_header or col_header:
        inner_height -= 2

    layout, cells_shown = grid_layout(arr, inner_width, inner_height)
    cell_width = grid_layout[0][0]
    label_formatter = StrFormatter()
    
    if arr.axis[1].ticks:
        # use one character less than available, to make labels more visually
        # separate

        col_label_layout = [label_formatter.format(str(label)[:cell_width-1], cell_width) for label in cells_shown.axis[1].ticks]
        layout = [col_label_layout] + layout

    if arr.axis[0].ticks:
        layout = [[' '*row_label_width] + row for row in layout]
        ticks = cells_shown.axis[0].ticks
        offset = 0
        if arr.axis[1].ticks: offset = 1
        for r in xrange(cells_shown.shape[0]):
            layout[r+offset][0] = label_formatter.format(ticks[r], row_label_width)
    
    return layout

def layout_to_string(layout):
    return '\n'.join([' '.join(row) for row in layout])

