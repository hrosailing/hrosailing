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
        """

class FillLocalImputator(Imputator):
    """
    An Imputator which fills missing data by ....
    """

    def __init__(
        self,
        fill_before=lambda name, right, mu: right,
        fill_between=lambda name, right, left, mu: right,
        fill_after=lambda name, left, mu: left,
        max_time_diff=timedelta(minutes=2)
    ):
        self._fill_before = fill_before
        self._fill_between = fill_between
        self._fill_after = fill_after
        self._max_time_diff = max_time_diff

    def imputate(
        self,
        data_dict
    ):
        n_removed_fields = len(data_dict)
        # remove all None fields
        data_dict = {key: value for key, value in data_dict.items()
                     if not all([v is None for v in value])}
        n_removed_fields -= len(data_dict)

        last_dt, last_i = None, None
        for i, dt in enumerate(data_dict["datetime"]):
            if dt is None:
                continue
            elif last_dt is None:
                pass
            elif abs(dt - last_dt) > self._max_time_diff:
                # remove all rows where datetime points are too far apart in time
                remove_rows.extend(list(range(last_i+1, i)))
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
        data_dict = \
            {key: [v for i, v in enumerate(value)
                   if i not in remove_rows]
             for key, value in data_dict.items()}
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
                self._fill_range_before(
                    data_dict,
                    datetime,
                    key,
                    start_idx,
                    idx[0]
                )

            # convex interpolation of entries between non-None entries
            for idx1, idx2 in zip(idx, idx[1:]):
                timediff = datetime[idx2] - datetime[idx1]
                if timediff < self._max_time_diff:
                    #fill data according to fill_between function
                    self._fill_range_between(
                        data_dict,
                        datetime,
                        key,
                        idx1,
                        idx2
                    )
                else:
                    #fill data according to fill_before and fill_after
                    near_points = \
                        [i for i in range(idx1 + 1, idx2)
                         if datetime[i] - datetime[idx1] < self._max_time_diff]
                    if len(near_points) > 0:
                        last_idx_right = max(near_points)
                        self._fill_range_after(
                            data_dict,
                            datetime,
                            key,
                            idx1,
                            last_idx_right
                        )
                    near_points = \
                        [i for i in range(idx1 + 1, idx2)
                         if datetime[idx2] - datetime[i] < self._max_time_diff]
                    if len(near_points) > 0:
                        first_idx_left = min(near_points)
                        self._fill_range_before(
                            data_dict,
                            datetime,
                            key,
                            first_idx_left,
                            idx2
                        )

            #fill last entries according to 'fill_after'
            near_points = \
                [i for i in range(idx[0])
                 if datetime[idx[0]] - datetime[i] < self._max_time_diff]
            if len(near_points) > 0:
                end_idx = min(near_points) #first idx in time interval
                self._fill_range_after(
                    data_dict,
                    datetime,
                    key,
                    start_idx,
                    end_idx
            )

        #remove rows which still have None values
        remove_rows = [i for i, (key, val) in enumerate(data_dict.items())
                       if any([v is None for v in val])]

        data_dict = \
            {key: [v for i, v in enumerate(value)
                   if i not in remove_rows]
             for key, value in data_dict.items()}
        n_removed_rows += len(remove_rows)

        statistics = {
            "n_removed_fields": n_removed_fields,
            "n_removed_rows": n_removed_rows,
        }

        return data_dict, statistics

    def _fill_range_between(
            self,
            data_dict,
            datetime,
            key,
            start_idx,
            end_idx
    ):
        left = data_dict[key][start_idx]
        right = data_dict[key][end_idx]
        for i in range(start_idx + 1, end_idx):
            duration = (datetime[end_idx] - datetime[start_idx])
            try:
                mu = (datetime[i] - datetime[end_idx])/duration
            except ZeroDivisionError:
                mu = 0
            data_dict[key][i] = self._fill_between(key, left, right, mu)

    def _fill_range_before(
            self,
            data_dict,
            datetime,
            key,
            start_idx,
            end_idx
    ):
        right = data_dict[key][end_idx]
        for i in range(start_idx + 1, end_idx):
            duration = (datetime[end_idx] - datetime[start_idx])
            try:
                mu = (datetime[i] - datetime[end_idx]) / duration
            except ZeroDivisionError:
                mu = 0
            data_dict[key][i] = self._fill_before(key, right, mu)

    def _fill_range_after(
            self,
            data_dict,
            datetime,
            key,
            start_idx,
            end_idx
    ):
        left = data_dict[key][start_idx]
        for i in range(start_idx + 1, end_idx):
            duration = (datetime[end_idx] - datetime[start_idx])
            try:
                mu = (datetime[i] - datetime[end_idx]) / duration
            except ZeroDivisionError:
                mu = 0
            data_dict[key][i] = self._fill_after(key, left, mu)
