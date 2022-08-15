import numpy as np

class Data:
    def __init__(self, data):
        self._data = {}
        self._types = {}
        self._weights = []
        self._max_len = 0

        self.update(data)

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

