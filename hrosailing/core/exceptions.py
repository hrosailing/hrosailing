"""Non standard exceptions"""


class FileReadingException(Exception):
    """Exception raised if non-OS-error occurs,
    when reading a file.
    """


class BilinearInterpolatorOutsideGridException(Exception):
    """Exception raised if bilinear interpolator
    tries interpolation outside the grid
    """


class BilinearInterpolatorNoGridException(Exception):
    """Exception raised if bilinear interpolator
    detects that values are not in a grid
    """
