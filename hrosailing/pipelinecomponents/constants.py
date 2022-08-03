"""
Contains constants used in the pipeline.
KEYSYNONYMS is a dictionary mapping commonly used sailing terms
(in lower case and the words are " " seperated)
to the hrosailing standard term.
"""

KEYSYNONYMS = {
    "wind": "TWS",
    "wind speed": "TWS",
    "true": "TWS",
    "true wind": "TWS",
    "true wind speed": "TWS",
    "ws": "TWS",
    "tws": "TWS",

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
    "lattitude": "lat",
    "lats": "lat",
    "lattitudes": "lat",

    "lon": "lon",
    "long": "lon",
    "longitude": "lon",
    "lons": "lon",
    "longs": "lon",
    "longitudes": "lon",
}
