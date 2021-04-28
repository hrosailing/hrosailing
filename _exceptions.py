# V: In Arbeit
class PolarDiagramException(Exception):
    """PolarDiagram exception class

    """
    def __init__(self, exception_type, *args):
        message_dict = {
            "Wrong dimension": "Expecting 2 dimensional array to be viewed as "
                               "Polar Diagram Tableau," +
                               f"\n got {args[0]} dimensional array instead.",
            "Wrong resolution": "Expecting resolution of type 'Iterable' or "
                                "'int/float'," +
                                f"\n got resolution of type {args[0]} instead",
            "Wrong shape": f"Expecting array with shape {args[0]},"
                           f"\n got array with shape {args[1]} instead",
            "Not in resolution": f"Expecting {args[0]} to be a subset"
                                 f" of {args[1]}",
            "No points found": f"The given true wind speed {args[0]} "
                               f"yielded no points in the current point cloud",
            "Not yet implemented": "Functionality is not yet implemented"
        }
        if exception_type in message_dict:
            super().__init__(message_dict[exception_type])
        else:
            super().__init__(exception_type)


class ProcessingException(Exception):
    pass
