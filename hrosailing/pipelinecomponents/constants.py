"""
Contains constants used in the pipeline.

- `KEYSYNONYMS` is a dictionary mapping commonly used sailing terms
(in lower case and the words are " " separated)
to the hrosailing standard term,

- `SEPARATORS` is a list containing characters that will be interpreted as separators between words when applying the
hrosailing standard format,

- `NORM_SCALES` is a dictionary containing suggested scale factors for various hrosailing standard keywords. These scales
are intended to be used to scale components before applying a norm,

- `HROSAILING_TO_NMEA` is a dictionary containing `key`, `value` pairs where `key` is a hrosailing standard keyword and
`value` is a list of NMEA 0183 keywords which are required for the computation of the data associated with `key`.
"""

KEYSYNONYMS = {
    "wind": "TWS",
    "wind speed": "TWS",
    "true": "TWS",
    "true wind": "TWS",
    "true wind speed": "TWS",
    "ws": "TWS",
    "tws": "TWS",
    "wspd": "TWS",
    "apparent wind speed": "AWS",
    "apparent wind": "AWS",
    "apparent": "AWS",
    "aws": "AWS",
    "angle": "TWA",
    "wind angle": "TWA",
    "true angle": "TWA",
    "true wind angle": "TWA",
    "wa": "TWA",
    "twa": "TWA",
    "apparent wind angle": "AWA",
    "apparent angle": "AWA",
    "awa": "AWA",
    "speed": "BSP",
    "boat speed": "BSP",
    "boat speed knots": "BSP",
    "vessel speed": "BSP",
    "vessel speed knots": "BSP",
    "sp": "BSP",
    "bsp": "BSP",
    "bsps": "BSP",
    "water speed": "BSP",
    "water speed knots": "BSP",
    "wsp": "BSP",
    "speed over ground": "SOG",
    "speed over ground knots": "SOG",
    "og": "SOG",
    "sog": "SOG",
    "spd over grnd": "SOG",
    "date": "date",
    "datestamp": "date",
    "date stamp": "date",
    "time": "time",
    "timestamp": "time",
    "time stamp": "time",
    "datetime": "datetime",
    "date time": "datetime",
    "lat": "lat",
    "latitude": "lat",
    "lon": "lon",
    "long": "lon",
    "longitude": "lon",
    "temp": "temp",
    "temperature": "temp",
    "dewpoint": "dewpoint",
    "dewpt": "dewpoint",
    "dwpt": "dewpoint",
    "dwpoint": "dewpoint",
    "rel humidity": "humidity",
    "relative humidity": "humidity",
    "rel hum": "humidity",
    "relative hum": "humidity",
    "rhum": "humidity",
    "relhum": "humidity",
    "hum": "humidity",
    "humidity": "humidity",
    "wd": "WD",
    "wdir": "WD",
    "winddir": "WD",
    "wpgt": "gust",
    "pres": "air pressure",
    "tsun": "total sunshine",
}


HROSAILING_TO_NMEA = {
    "TWS": ["Wind speed", "Reference"],
    "AWS": ["Wind speed", "Reference"],
    "TWA": ["Wind angle", "Reference"],
    "AWA": ["Wind angle", "Reference"],
    "lat": ["Latitude"],
    "lon": ["Longitude"],
    "SOG": ["Speed Over Ground"],
    "date": ["Datestamp"],
    "time": ["Timestamp"],
}

SEPARATORS = [",", ";", "_", "-", "+", "&", "\n", "\t"]

NORM_SCALES = {
    "TWS": 1 / 20,
    "AWS": 1 / 20,
    "TWA": 1 / 360,
    "AWA": 1 / 360,
    "BSP": 1 / 40,
    "SOG": 1 / 40,
    "lat": 1 / 360,
    "lon": 1 / 360,
}
