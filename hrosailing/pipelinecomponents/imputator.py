"""
Contains the Base class for the Imputator pipeline-component that can be used
to create custom Imputators.
"""

from abc import ABC, abstractmethod
from datetime import timedelta


class Imputator(ABC):
    """Base class for all imputator classes


    Abstract Methods
    ----------------
    imputate(self, data)
    """

    @abstractmethod
    def imputate(self, data):
        """
        This method should be used, given data that possibly contains None
        values to create data without any None values

        Parameter
        --------
        data: Data,
            data to be imputated
        """

class FillLocalImputator(Imputator):
    """
    An Imputator which assumes that the data has been stored chronologically
    and contains some datestamps.
    Fills missing data by:
    - deleting columns that only contain None values
    - delete rows without datetime-stamp which lie between two data_points
        which are far apart in time, before the first or after the last
        datestamp
    - affine interpolation of datetime-stamps between two data_points which
        are not far apart in time
    - fill data before and after a not-None value which are not too far apart
        in time according to certain functions ('fill_before' and 'fill_after)
    - fill data between a pair of not-None values which are not too far apart
        in time according to a certain function ('fill_between')

    Parameters:
    -----------------
    fill_before: (str, object, float) -> object, optional
        The function which will be used to fill None fields before a not-None
        field.

        First argument is the attribute name, second is the value of the
        not_None field mentioned before, third argument is the relative
        position (in time) between the earliest applicable data and the
        mentioned not-None data. Returns the value to be filled

        Defaults to 'lambda name, right, mu: right'

    fill_between: (str, object, object, float) -> object
        The function which will be used to fill None fields between
        two not-None field.

        First argument is the attribute name, second and third are the values
        of the not_None fields mentioned before in chronological order,
        last argument is the relative position (in time) between
        the the two mentioned not-None data points.
        Returns the value to be filled

        Defaults to 'lambda name, left, right, mu: left'

    fill_after: (str, object, float) -> object, optional
        The function which will be used to fill None fields after a not-None
        field.

        First argument is the attribute name, second is the value of the
        not_None field mentioned before, third argument is the relative
        position (in time) between the mentioned not-None data and the
        latest applicable data. Returns the value to be filled

        Defaults to 'lambda name, left, mu: left'

    max_time_diff: datetime.timedelta, optional
        Two data points are treated as 'close in time' if their time difference
        is smaller than 'max_time_diff'

        Defaults to 2 minutes
    """

    def __init__(
        self,
        fill_before=lambda name, right, mu: right,
        fill_between=lambda name, left, right, mu: left,
        fill_after=lambda name, left, mu: left,
        max_time_diff=timedelta(minutes=2)
    ):
        self._fill_before = \
            lambda name, left, right, mu: fill_before(name, right, mu)
        self._fill_between = fill_between
        self._fill_after = \
            lambda name, left, right, mu: fill_after(name, left, mu)
        self._max_time_diff = max_time_diff
        self._n_filled = 0

    def imputate(
        self,
        data_dict
    ):
        """
        Creates a dictionary that does not contain None values by the method
        described above.

        Parameters
        ----------------
        data_dict: dict
            The dictionary to be imputated

        Returns
        -------------
        data_dict, statistics: dict, dict
            data_dict is the resulting dictionary containing no None values
            statistics contains the number of removed columns, the number of
            removed rows, the number of filled fields and the number of rows and
            columns in the resulting dictionary as
            "n_removed_cols", "n_removed_rows", "n_filled_fields",
            "n_rows" and "n_cols" respectively
        """
        self._n_filled = 0
        n_removed_cols = data_dict.n_cols
        data_dict.strip("cols")
        n_removed_cols -= data_dict.n_cols

        last_dt, last_i = None, None
        for i, dt in enumerate(data_dict["datetime"]):
            if dt is None:
                continue
            elif last_dt is None:
                pass
            elif abs(dt - last_dt) > self._max_time_diff:
                continue
            else:
                # linear approximation of time in between
                for j in range(last_i+1, i):
                    mu = (j - last_i)/(i - last_i)
                    data_dict["datetime"][j] = last_dt + mu*(dt - last_dt)
            last_dt, last_i = dt, i
        remove_rows = [
            i for i, dt in enumerate(data_dict["datetime"])
            if dt is None
        ]
        data_dict.delete(remove_rows)

        n_removed_rows = len(remove_rows)

        # indices of not None values
        idx_dict = {
            key: [i for i, data in enumerate(data_dict[key]) if
                  data is not None]
            for key in data_dict
        }

        datetime = data_dict["datetime"]

        for key, idx in idx_dict.items():
            # fill every entry before the first not-None entry according to the
            # "fill before" function
            if key == "datetime":
                continue
            if idx[0] > 0:
                start_idx = min(
                    [i for i in range(idx[0])
                     if datetime[idx[0]] - datetime[i] < self._max_time_diff]
                ) #first idx in time interval
                self._fill_range(
                    data_dict,
                    datetime,
                    key,
                    start_idx,
                    idx[0],
                    self._fill_before
                )

            # convex interpolation of entries between non-None entries
            for idx1, idx2 in zip(idx, idx[1:]):
                timediff = datetime[idx2] - datetime[idx1]
                if timediff < self._max_time_diff:
                    #fill data according to fill_between function
                    self._fill_range(
                        data_dict,
                        datetime,
                        key,
                        idx1,
                        idx2,
                        self._fill_between
                    )
                else:
                    #fill data according to fill_before and fill_after
                    near_points = \
                        [i for i in range(idx1 + 1, idx2)
                         if datetime[i] - datetime[idx1] < self._max_time_diff]
                    if len(near_points) > 0:
                        last_idx_right = max(near_points)
                        self._fill_range(
                            data_dict,
                            datetime,
                            key,
                            idx1,
                            last_idx_right,
                            self._fill_after
                        )
                    near_points = \
                        [i for i in range(idx1 + 1, idx2)
                         if datetime[idx2] - datetime[i] < self._max_time_diff]
                    if len(near_points) > 0:
                        first_idx_left = min(near_points)
                        self._fill_range(
                            data_dict,
                            datetime,
                            key,
                            first_idx_left,
                            idx2,
                            self._fill_before
                        )

            #fill last entries according to 'fill_after'
            near_points = \
                [i for i in range(idx[0])
                 if datetime[idx[0]] - datetime[i] < self._max_time_diff]
            if len(near_points) > 0:
                end_idx = min(near_points) #first idx in time interval
                self._fill_range(
                    data_dict,
                    datetime,
                    key,
                    start_idx,
                    end_idx,
                    self._fill_after
            )

        #remove rows which still have None values
        remove_rows = [i for i, _ in enumerate(data_dict["datetime"])
                       if any([data_dict[key][i] is None
                               for key in data_dict.keys()])]

        data_dict = \
            {key: [v for i, v in enumerate(value)
                   if i not in remove_rows]
             for key, value in data_dict.items()}
        n_removed_rows += len(remove_rows)

        statistics = {
            "n_removed_cols": n_removed_cols,
            "n_removed_rows": n_removed_rows,
            "n_filled_fields": self._n_filled,
            "n_rows": len(list(data_dict.values())[0]),
            "n_cols": len(data_dict)
        }

        return data_dict, statistics

    def _fill_range(
            self,
            data_dict,
            datetime,
            key,
            start_idx,
            end_idx,
            fill_fun
    ):
        left = data_dict[key][start_idx]
        right = data_dict[key][end_idx]
        for i in range(start_idx + 1, end_idx):
            duration = (datetime[end_idx] - datetime[start_idx])
            try:
                mu = (datetime[i] - datetime[end_idx])/duration
            except ZeroDivisionError:
                mu = 0
            data_dict[key][i] = fill_fun(key, left, right, mu)
            self._n_filled += 1
