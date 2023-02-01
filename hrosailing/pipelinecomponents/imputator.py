"""
Contains the Base class for the `Imputator` pipeline-component that can be used
to create custom imputators.
"""

from abc import ABC, abstractmethod
from datetime import timedelta

from hrosailing.pipelinecomponents._utils import ComponentWithStatistics


class Imputator(ComponentWithStatistics, ABC):
    """Base class for all imputator classes.


    Abstract Methods
    ----------------
    impute(self, data)
    """

    @abstractmethod
    def impute(self, data):
        """
        This method should be used, given data that possibly contains `None`
        values, to create data without any `None` values.

        Parameters
        ----------
        data : Data
            Data to be imputed.
        """

    def set_statistics(
        self, n_removed_cols, n_removed_rows, n_filled, data_dict
    ):
        super().set_statistics(
            n_removed_cols=n_removed_cols,
            n_removed_rows=n_removed_rows,
            n_filled_fields=n_filled,
            n_rows=data_dict.n_rows,
            n_cols=data_dict.n_cols,
        )


class RemoveOnlyImputator(Imputator):
    """
    Imputator that removes all entries containing None values.
    """

    def impute(self, data):
        """
        Removes all entries containing None values.

        See also
        --------
        `Imputator.impute`
        """
        n_rows = data.n_rows
        remove_rows = [
            idx
            for idx, row in enumerate(data.rows(return_type=list))
            if None in row
        ]
        data.delete(remove_rows)

        self.set_statistics(0, n_rows - len(remove_rows), 0, data)

        return data


class FillLocalImputator(Imputator):
    """
    An `Imputator` which assumes that the data has been stored chronologically
    and contains the field 'datetime'.
    Fills missing data by:
    - deleting columns that only contain `None` values,
    - deleting rows between two data-points that are far apart in time,
    - affine interpolation of datetime-stamps between two data-points which
        are not far apart in time,
    - filling data before and after a not-`None` value which are not too far apart
        in time according to certain functions (`fill_before` and `fill_after`),
    - filling data between a pair of not-`None` values which are not too far apart
        in time according to a certain function (`fill_between`).

    Parameters
    ----------
    fill_before : (str, object, float) -> object, optional
        The function which will be used to fill `None` fields before a not-`None`
        field.

        First argument is the attribute name, second is the value of the
        not-`None` field mentioned before, third argument is the relative
        position (in time) between the earliest applicable data and the
        mentioned not-`None` data. Returns the value that replaces `None`.

        Defaults to `lambda name, right, mu: right`.

    fill_between : (str, object, object, float) -> object
        The function which will be used to fill `None` fields between
        two not-`None` fields.

        First argument is the attribute `name`, second and third are the values
        of the not-`None` fields mentioned before in chronological order,
        last argument is the relative position (in time) between
        the two mentioned not-`None` data points.
        Returns the value to be filled.

        Defaults to `lambda name, left, right, mu: left`.

    fill_after : (str, object, float) -> object, optional
        The function which will be used to fill `None` fields after a not-`None`
        field.

        First argument is the attribute `name`, second is the value of the
        not-`None` field mentioned before, third argument is the relative
        position (in time) between the mentioned not-`None` data and the
        latest applicable data. Returns the value to be filled.

        Defaults to `lambda name, left, mu: left`.

    max_time_diff : datetime.timedelta, optional
        Two data points are treated as 'close in time' if their time difference
        is smaller than `max_time_diff`.

        Defaults to `2 minutes`.
    """

    def __init__(
        self,
        fill_before=lambda name, right, mu: right,
        fill_between=lambda name, left, right, mu: left,
        fill_after=lambda name, left, mu: left,
        max_time_diff=timedelta(minutes=2),
    ):
        super().__init__()
        self._fill_before = lambda name, left, right, mu: fill_before(
            name, right, mu
        )
        self._fill_between = fill_between
        self._fill_after = lambda name, left, right, mu: fill_after(
            name, left, mu
        )
        self._max_time_diff = max_time_diff
        self._n_filled = 0

    def impute(self, data):
        """
        Creates a dictionary that does not contain `None` values by the procedure
        described above.

        Parameters
        ----------
        data : Data
            The `Data` object to be imputed.

        Returns
        -------
        data_dict : Data
            `data_dict` is the resulting `Data` object containing no `None` values.
        """
        self._n_filled = 0
        n_removed_cols = data.n_cols
        data.strip("cols")
        n_removed_cols -= data.n_cols

        data = self._interpolate_datetime(data)

        data, n_removed_rows = self._remove_rows(data)

        data = self._interpolate_other(data)

        # remove rows which still have None values
        remove_rows = [
            i
            for i, _ in enumerate(data["datetime"])
            if any(data[key][i] is None for key in data.keys())
        ]

        data.delete(remove_rows)
        n_removed_rows += len(remove_rows)

        self.set_statistics(
            n_removed_cols, n_removed_rows, self._n_filled, data
        )

        return data

    def _fill_range(
        self, data_dict, datetime, key, start_idx, end_idx, fill_fun
    ):
        left = data_dict[key][start_idx]
        right = data_dict[key][end_idx]
        for i in range(start_idx + 1, end_idx):
            duration = datetime[end_idx] - datetime[start_idx]
            try:
                mu = (datetime[i] - datetime[start_idx]) / duration
            except ZeroDivisionError:
                mu = 0
            data_dict[key][i] = fill_fun(key, left, right, mu)
            self._n_filled += 1

    def _interpolate_datetime(self, data):
        last_dt, last_i = None, None
        for i, dt in enumerate(data["datetime"]):
            if dt is None:
                continue
            if last_dt is None:
                pass
            elif abs(dt - last_dt) > self._max_time_diff:
                continue
            else:
                # linear approximation of time in between
                for j in range(last_i + 1, i):
                    mu = (j - last_i) / (i - last_i)
                    data["datetime"][j] = last_dt + mu * (dt - last_dt)
            last_dt, last_i = dt, i

        return data

    def _remove_rows(self, data):
        remove_rows = [
            i for i, dt in enumerate(data["datetime"]) if dt is None
        ]
        data.delete(remove_rows)

        n_removed_rows = len(remove_rows)

        return data, n_removed_rows

    def _interpolate_other(self, data):
        # indices of not None values
        idx_dict = {
            key: [i for i, data in enumerate(data[key]) if data is not None]
            for key in data.keys()
        }

        datetime = data["datetime"]

        for key, idx in idx_dict.items():
            # fill every entry before the first not-None entry according to the
            # "fill before" function
            if key == "datetime":
                continue

            try:
                start_idx = self._aplly_fill_before(datetime, idx, key, data)
            except ValueError:
                continue

            self._apply_fill_between(key, idx, data, datetime)

            self._apply_fill_after(key, idx, data, datetime, start_idx)

        return data

    def _aplly_fill_before(self, datetime, idx, key, data):
        if idx[0] > 0:
            start_idx = min(
                i
                for i in range(idx[0])
                if datetime[idx[0]] - datetime[i] < self._max_time_diff
            )  # first idx in time interval

            self._fill_range(
                data,
                datetime,
                key,
                start_idx,
                idx[0],
                self._fill_before,
            )

            return start_idx
        return None

    def _apply_fill_between(self, key, idx, data, datetime):
        for idx1, idx2 in zip(idx, idx[1:]):
            timediff = datetime[idx2] - datetime[idx1]
            if timediff < self._max_time_diff:
                # fill data according to fill_between function
                self._fill_range(
                    data,
                    datetime,
                    key,
                    idx1,
                    idx2,
                    self._fill_between,
                )
            else:
                # fill data according to fill_before and fill_after
                near_points = [
                    i
                    for i in range(idx1 + 1, idx2)
                    if datetime[i] - datetime[idx1] < self._max_time_diff
                ]
                if len(near_points) > 0:
                    last_idx_right = max(near_points)
                    self._fill_range(
                        data,
                        datetime,
                        key,
                        idx1,
                        last_idx_right,
                        self._fill_after,
                    )
                near_points = [
                    i
                    for i in range(idx1 + 1, idx2)
                    if datetime[idx2] - datetime[i] < self._max_time_diff
                ]
                if len(near_points) > 0:
                    first_idx_left = min(near_points)
                    self._fill_range(
                        data,
                        datetime,
                        key,
                        first_idx_left,
                        idx2,
                        self._fill_before,
                    )

    def _apply_fill_after(self, key, idx, data, datetime, start_idx):
        near_points = [
            i
            for i in range(idx[0])
            if datetime[idx[0]] - datetime[i] < self._max_time_diff
        ]
        if len(near_points) > 0:
            end_idx = min(near_points)  # first idx in time interval
            self._fill_range(
                data,
                datetime,
                key,
                start_idx,
                end_idx,
                self._fill_after,
            )
