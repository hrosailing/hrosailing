"""Contains the class `Data` which is an output of several pipeline components.
"""

from datetime import datetime
from decimal import Decimal

import numpy as np

from hrosailing.pipelinecomponents.constants import KEYSYNONYMS, SEPARATORS


class Data:
    """
    Organizes measurement data. Data is interpreted as a table with columns
    identified by keywords and rows corresponding to a certain measurement
    point. Also saves and checks the data types of the corresponding columns.
    Weights might be added to the corresponding rows.

    If `data` is of type `Data`, `data[key]` yields the corresponding column if
    `key` is a string and the corresponding row if `key` is an integer.
    `key in data` checks if there is a column with key `key`.
    Iteration is performed over the rows.

    Properties
    ----------
    numerical : numpy.ndarray
        An array containing all data of type `float`.

    n_rows : int
        The number of rows.

    n_cols : int
        The number of columns.
    """

    def __init__(self):
        self._data = {}
        self._types = {}
        self._max_len = 0

    def keys(self):
        """
        Returns
        -------
        keys : list of str
            List of all column names.
        """
        return list(self._data.keys())

    @property
    def numerical(self):
        float_keys, float_vals = self.get_by_type(float)
        array = np.column_stack(float_vals)
        return float_keys, array

    @property
    def n_rows(self):
        return self._max_len

    @property
    def n_cols(self):
        return len(self._data)

    @property
    def data(self):
        return self._data.copy()

    def rows(self, keys=None, return_type=dict):
        """
        Iterates over the rows.

        Parameters
        ----------
        keys : iterable over str, optional
            The columns from which the iterator yields. Other columns will be ignored.

            Defaults to `self.keys()`.

        return_type : dict, tuple or list
            Defines the type of the yielded data. If types `tuple` or `list` are used, the entries are in the exact same
            order as in `keys`.

            Defaults to `dict`.
        """
        if keys is None:
            keys = self.keys()
        for i in range(self._max_len):
            if return_type is dict:
                yield {key: self._data[key][i] for key in keys}
            elif return_type is tuple:
                yield (self._data[key][i] for key in keys)
            elif return_type is list:
                yield [self._data[key][i] for key in keys]
            else:
                raise ValueError(
                    f"Return type of {return_type} is not supported"
                )

    def get_by_type(self, type_):
        """
        Parameters
        ----------
        type_ : type

        Returns
        -------
        filtered_keys : list of str
            A list of all keys with columns of type `type_`.

        filtered_data : list of lists
            A list of all data corresponding to the columns with keys
            `filtered_keys`.
        """
        filtered_keys = [
            key for key in self.keys() if self._types[key] == type_
        ]
        return filtered_keys, [self._data[key] for key in filtered_keys]

    def type(self, key):
        """
        Returns the associated type corresponding to the `key` column.

        Parameters
        ----------
        key : str
        """
        return self._types[key]

    def extend(self, key, data):
        """
        Extends the data by given other data.

        Parameters
        ----------
        key : str
            The key of the column that should be extended by the given data.
        data : list
            Data that should be appended to the given column.
        """
        data_type = self._get_type(data)
        if key in self._data:
            if data_type is not None and data_type != self._types[key]:
                raise ValueError(
                    f"data should be of type {self._types[key]}"
                    f" but has type {data_type}"
                )
            self._data[key].extend(data)
            self._max_len = max(self._max_len, len(self._data[key]))
        else:
            self.fill(self._max_len - len(data), [key])
            self._data[key].extend(data)
            self._types[key] = data_type
            self._max_len = max(self._max_len, len(self._data[key]))

    def update(self, data_dict):
        """
        Extends the data according to given data and fills missing
        entries in each column with `None`.

        Parameters
        ----------
        data_dict : dict or Data
            The dictionary or `Data` object containing the data to be used for the update.
        """
        if isinstance(data_dict, dict):
            for key, val in data_dict.items():
                if isinstance(val, list):
                    self.extend(key, val)
                else:
                    self.append(key, val)
            self.fill()
        # if isinstance(data_dict, Data): this does not work for some reason
        else:
            self.update(data_dict.data)

    def append(self, key, data):
        """
        Extends the data by a single element.

        Parameters
        ----------
        key : str
            The key of the column to which the given data should be appended.
        data : object
            Single data object to be appended.

        See also
        --------
        `Data.extend`
        """
        self.extend(key, [data])

    def fill(self, len_=None, keys=None):
        """
        Fills all columns smaller than a specific length with `None` values.

        Parameters
        ----------
        len_ : int, optional
            The required length.

            Defaults to `self.n_rows`.

        keys : list of str, optional
            The keys of the columns which should be filled.

            Defaults to `self.keys`.
        """
        if keys is None:
            keys = self.keys()
        if len_ is None:
            len_ = self._max_len
        for key in keys:
            if key not in self._data:
                self._data[key] = []
            curr_len = len(self._data[key])
            fill_len = max(0, (len_ - curr_len))
            self._data[key].extend([None] * fill_len)
            self._types[key] = self._get_type(self._data[key])

    def filter_types(self, type_list):
        """
        Deletes all data which is not in a list of required types.

        Parameters
        ----------
        type_list : list of types
        """
        for key in list(self._data.keys()):
            if self._types[key] not in type_list:
                del self._data[key]
                del self._types[key]

    def rename(self, old_key, new_key):
        """
        Renames a column.

        Parameters
        ----------
        old_key : str
            Old name of column.

        new_key : str
            New name of column.
        """
        if old_key == new_key:
            return
        if new_key in self.keys():
            raise ValueError(
                f"Can not rename {old_key} to {new_key} since"
                f" {new_key} already exists"
            )

        self._data[new_key] = self._data[old_key]
        self._types[new_key] = self._types[old_key]

        self.delete(old_key)

    def delete(self, key):
        """
        Deletes a column or a row.

        Parameters
        ----------
        key : int, str, list of str or list of int
            Name(s) of column(s) to be deleted or index(es) of row(s) to be deleted.
        """
        if isinstance(key, str):
            del self._data[key]
            del self._types[key]
        if isinstance(key, int):
            self.delete([key])
        if isinstance(key, list):
            if len(key) == 0:
                return
            if isinstance(key[0], str):
                for k in key:
                    self.delete(k)
            if isinstance(key[0], int):
                self._data = {
                    k: [v for i, v in enumerate(value) if i not in key]
                    for k, value in self._data.items()
                }
                self._max_len -= len(key)

    def strip(self, mode="all"):
        """
        Removes either columns that only consist of `None` values or leading and tailing rows consisting only of `None`
        values.

        Parameters
        ----------
        mode : {"cols", "rows", "all"}
            Specifies whether columns or rows or both should be removed.

            Defaults to "all"
        """
        if mode == "cols":
            self._data = {
                key: value
                for key, value in self._data.items()
                if not all(v is None for v in value)
            }
        elif mode == "rows":
            end_leading_nones = 0  # first index AFTER leading None rows
            start_tailing_nones = (
                self._max_len - 1
            )  # first index BEFORE tailing None rows
            while all(
                value[end_leading_nones] is None
                for key, value in self._data.items()
            ):
                end_leading_nones += 1
            while all(
                value[start_tailing_nones] is None
                for key, value in self._data.items()
            ):
                start_tailing_nones -= 1
            self._data = {
                key: value[end_leading_nones : start_tailing_nones + 1]
                for key, value in self._data.items()
            }
        elif mode == "all":
            self.strip("cols")
            self.strip("rows")

    def hrosailing_standard_format(self):
        """
        Reformats data into the `hrosailing` standard format.

        This means:
            - the dictionary has `hrosailing` standard keys whenever possible,
            - date and time fields will be aggregated to datetime,
            - tries to cast entries to `float` whenever possible.
        """

        def get_standard_key(key):
            lkey = key.lower()
            for sep in SEPARATORS:
                lkey = lkey.replace(sep, " ")
            lkey = lkey.strip()
            # remove plural
            stripped_lkey = lkey if lkey[-1] != "s" else lkey[:-1]
            if lkey in KEYSYNONYMS:
                return KEYSYNONYMS[lkey]
            if stripped_lkey in KEYSYNONYMS:
                return KEYSYNONYMS[stripped_lkey]

            return key

        for key, value in list(self._data.items()):
            standard_key = get_standard_key(key)
            new_key = standard_key
            if key == new_key:
                continue
            i = 0
            while new_key in self.keys():
                i += 1
                new_key = f"{standard_key}(Control value {i})"
            self.rename(key, new_key)

        if "time" in self and "date" in self:

            def combine(date, time):
                if date is None or time is None:
                    return None
                return datetime.combine(date, time)

            self.extend(
                "datetime",
                [
                    combine(date, time)
                    for date, time in zip(self["date"], self["time"])
                ],
            )
            self.delete("date")
            self.delete("time")

        # ensure floats
        for key in self.keys():
            if self._types[key] in (int, str, Decimal, np.float_):
                succes, self._data[key] = _try_call_to_float(self._data[key])
                if succes:
                    self._types[key] = float

    @staticmethod
    def _get_type(data):
        curr_type = None
        for entry in data:
            if entry is None:
                continue
            if curr_type is None:
                curr_type = type(entry)
                continue
            if type(entry) != curr_type:
                raise ValueError(
                    "Data has no consistent type."
                    f"Found the types {type(entry)} and {curr_type}"
                )
        return curr_type

    @classmethod
    def concatenate(cls, list_):
        """
        Returns concatenated data, given a list of `Data` classes.

        Parameters
        ----------
        list_ : list of Data

        Returns
        -------
        data : Data
            The concatenated Data.
        """
        data = cls()
        for other_data in list_:
            data.update(other_data)
        return data

    @classmethod
    def _force_set(cls, data, types, max_len):
        # use with caution: wrong usage can produce inconsistent `Data` objects.
        new_obj = cls()
        new_obj._data = data
        new_obj._types = types
        new_obj._max_len = max_len
        return new_obj

    @classmethod
    def from_dict(cls, dict_):
        """
        Creates a `Data` object that contains the same data as a given dictionary.
        The keys and values of the dictionary should be iterables and will correspond to the columns of the resulting
        `Data` object.

        Parameters
        ----------
        dict_ : dict
        """
        data = cls()
        data.update(dict_)
        return data

    def get_slice(self, slice_):
        """
        Returns new `Data` object containing only keys in `slice`.

        Parameters
        ----------
        slice_ : object with method `__contains__`

        Returns
        -------
        data: Data
        """

        data = {
            key: value for key, value in self._data.items() if key in slice_
        }
        types = {
            key: value for key, value in self._types.items() if key in slice_
        }
        try:
            max_len = max(len(field) for field in data.values())
        except ValueError:
            max_len = 0
        return Data._force_set(data, types, max_len)

    def _mask_rows(self, mask):
        """
        Keeps only a subset of the rows indicated by `mask`.

        Parameters
        ----------
        mask : array_like of booleans
        """

        data = {
            key: [entry for entry, choose in zip(value, mask) if choose]
            for key, value in self._data.items()
        }
        try:
            max_len = max(len(field) for field in data.values())
        except ValueError:
            max_len = 0

        return Data._force_set(data, self._types.copy(), max_len)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._data[item]
        if isinstance(item, int):
            return {key: val[item] for key, val in self._data.items()}
        type_ = None
        for key in item:
            type_ = type(key)
            break
        if type_ is None or type_ in (bool, np.bool_):
            return self._mask_rows(item)
        if type_ is str:
            return self.get_slice(item)
        raise TypeError(
            "Only types `int`, `str` and iterables over `bool`, `numpy.bool_`"
            " or `str` are supported"
        )

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        for i in range(self._max_len):
            yield self[i]

    def __str__(self):
        str_ = ""
        for key in self.keys():
            str_ += f"Data field '{key}' of type {self._types[key]}:\n\t"
            if len(self._data[key]) < 10:
                str_ += str(self._data[key])
            else:
                str_ += (
                    f"[{','.join([str(item) for item in self._data[key][:5]])},"
                    " ..."
                    f" ,{','.join([str(item) for item in self._data[key][-5:]])}]"
                )
            str_ += "\n"

        return str_


def _try_call_to_float(list_):
    new_list = []
    for i, value in enumerate(list_):
        if value is None:
            new_list.append(value)
            continue
        try:
            new_list.append(float(value))
        except (TypeError, ValueError):
            return False, list_
    return True, new_list
