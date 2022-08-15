import numpy as np
from datetime import datetime

from hrosailing.pipelinecomponents.constants import KEYSYNONYMS, SEPERATORS


class Data:
    def __init__(self):
        self._data = {}
        self._types = {}
        self._weights = []
        self._max_len = 0

    @property
    def keys(self):
        return list(self._data.keys())

    @property
    def numerical(self):
        float_keys, float_vals = self.get_by_type(float)
        array = np.column_stack(float_vals)
        return float_keys, array

    def get_by_type(self, type_):
        filtered_keys = [
            key for key in self.keys if self._types[key] == type_
        ]
        return filtered_keys, [
            self._data[key] for key in filtered_keys
        ]

    def extend(self, key, data):
        data_type = self._get_type(data)
        if key in self._data:
            if data_type != self._types[key]:
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
        for key, val in data_dict.items():
            self.extend(key, val)
        self.fill()

    def append(self, key, data):
        self.extend(key, [data])

    def fill(self, len_=None, keys=None):
        if keys is None:
            keys = self.keys
        if len_ is None:
            len_ = self._max_len
        for key in keys:
            if key not in self._data:
                self._data[key] = []
            curr_len = len(self._data[key])
            fill_len = max(0, (len_ - curr_len))
            self._data[key].extend([None]*fill_len)

    def filter_types(self, type_list):
        for key in self._data.keys():
            if self._types[key] not in type_list:
                del self._data[key]
                del self._types[key]

    def rename(self, old_key, new_key):
        if old_key == new_key:
            return
        self._data[new_key] = self._data[old_key]
        self._types[new_key] = self._types[old_key]

        self.delete(old_key)

    def delete(self, key):
        del self._data[key]
        del self._types[key]

    def hrosailing_standard_format(self):
        """
            Reformats data in the hrosailing standard format.

            This means:
                - the dictionary has hrosailing standard keys whenever possible
                - date and time fields will be aggregated to datetime
                - tries to cast entries to `float` whenever possible
        """
        def standard_key(key):
            lkey = key.lower()
            for sep in SEPERATORS:
                lkey = lkey.replace(sep, " ")
            lkey = lkey.strip()
            return KEYSYNONYMS[lkey] if lkey in KEYSYNONYMS else key

        for key, value in self._data.items():
            self.rename(key, standard_key(key))

        if "time" in self and "date" in self:

            def combine(date, time):
                if date is None or time is None:
                    return None
                else:
                    return datetime.combine(date, time)

            self.extend(
                "datetime",
                [
                combine(date, time)
                for date, time in
                zip(self["date"], self["time"])
                ]
            )
            self.delete("date")
            self.delete("time")

    @staticmethod
    def _get_type(data):
        curr_type = None
        for field in data:
            if field is None:
                continue
            if curr_type is None:
                curr_type = type(field)
                continue
            if type(field) != curr_type:
                raise ValueError(
                    f"Data has no consistent type."
                    f"Found {type(field)} and {curr_type}"
                )
        return curr_type

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._data[item]
        if isinstance(item, int):
            return {key: val[item] for key, val in self._data.items()}

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        for i in range(self._max_len):
            yield self[i]
