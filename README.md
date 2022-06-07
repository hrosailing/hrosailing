[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![linter](https://github.com/hrosailing/hrosailing/actions/workflows/linting.yml/badge.svg)](https://github.com/hrosailing/hrosailing/actions/workflows/linting.yml)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![tester](https://github.com/hrosailing/hrosailing/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/hrosailing/hrosailing/actions/workflows/build-and-test.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/hrosailing/hrosailing/badge)](https://www.codefactor.io/repository/github/hrosailing/hrosailing)

<p align="center">
    <img src="https://raw.githubusercontent.com/hrosailing/hrosailing/main/logo.png" width=300px height=300px alt="hrosailing">
</p>

hrosailing 
==========
![Still in active development]!
---------------------------

The `hrosailing` package provides various tools and interfaces to
visualize, create and work with polar (performance) diagrams.

The main interface being the `PolarDiagram` interface for 
the creation of custom polar diagrams, which is compatible with
the functionalities of this package. `hrosailing` also provides some
pre-implemented classes inheriting from `PolarDiagram` which can be used as well.

The package contains a data processing framework, centered around the
`PolarPipeline` class, to generate polar diagrams from raw data. 

`pipelinecomponents` provides many out of the box parts for
the aforementioned framework as well as the possibility to easily
create own ones. 

The package also provides many navigational usages of polar
(performance) diagrams with `cruising`.

You can find the documentation [here](https://hrosailing.github.io/hrosailing/ "hrosailing").
See also the examples below for some showcases.


### Installation
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyPI version](https://badge.fury.io/py/hrosailing.svg)](https://badge.fury.io/py/hrosailing)

The recommended way to install `hrosailing` is with 
[pip](http://pypi.python.org/pypi/pip/):
    
    pip install hrosailing

The `hrosailing` package might also be compatible (in large) with 
earlier versions of Python, together with some earlier versions of some 
of the used packages, namely `numpy`, `scipy`, and `matplotlib`.


### Examples
In the following we showcase some of the capabilities of `hrosailing`.
All definitions of an example code might be used in the succeeding examples.


#### Serialization of `PolarDiagram` objects
For a first example, lets say we obtained some table with polar 
performance diagram data, like the one available 
[here](https://www.seapilot.com/wp-content/uploads/2018/05/60ftmono.txt), 
and call the file testdata.csv.

```python
import hrosailing.polardiagram as pol
# the format of `testdata.csv` is a tab separated one
# supported by the keyword `array`
pd = pol.from_csv("testdata.csv", fmt="array")

# for symmetric results
pd = pd.symmetrize()

# serialized the polar diagram to a .csv file
# in the style of an intern format
pd.to_csv("polar_diagram.csv")
# the default format is the intern format `hro`
pd2 = pol.from_csv("polar_diagram.csv")
```

Currently serialization is only supported for some csv-format, see also
[csv-format-examples](https://github.com/hrosailing/hrosailing/tree/main/examples/csv-format-examples)
for example files for the currently supported format. See also 
[Issue #1](https://github.com/hrosailing/hrosailing/issues/1) for a plan
to add more serialization options.


#### Visualizing polar diagrams
```python
import matplotlib.pyplot as plt

ws = [10, 20, 30]

pd.plot_polar(ws=ws, ax=plt.subplot(2, 2, 1, projection="polar"))
pd.plot_convex_hull(ws=ws, ax=plt.subplot(2, 2, 2, projection="polar"))
pd.plot_flat(ws=ws, ax=plt.subplot(2, 2, 3))
pd.plot_color_gradient(ax=plt.subplot(2, 2, 4))

plt.show()
```
![flat_plots](https://user-images.githubusercontent.com/70914876/146026223-fc58a914-9b01-47ae-bf9c-6429113dbf4a.png)

3d visualization is also supported.
```python
pd.plot_3d()
plt.show()
```
![3d_plot](https://user-images.githubusercontent.com/70914876/146153719-826e8c93-09ab-4387-b13c-e942139fcce6.png)


#### Iterate over polar diagram data
We can also directly iterate and/or evaluate the wind angles, 
wind speeds and boat speeds of the polar diagram.

```python
import numpy as np


def random_shifted_pt(pt, mul):
    pt = np.array(pt)
    rand = np.random.random(pt.shape) - 0.5
    rand *= np.array(mul)
    return pt + rand


data = np.array([
    random_shifted_pt([ws, wa, pd(ws, wa)[0]], [10, 5, 2])
    for wa in pd.wind_angles
    for ws in pd.wind_speeds
    for _ in range(6)
])
data = data[np.random.choice(len(data), size=500)]
```


#### Creating polar diagrams from raw data
```python
import hrosailing.pipeline as pipe
import hrosailing.pipelinecomponents as pcomp


pol_pips = [
    pipe.PolarPipeline(
        handler=pcomp.ArrayHandler(),
        extension=pipe.TableExtension()
    ),
    pipe.PolarPipeline(
        handler=pcomp.ArrayHandler(),
        extension=pipe.PointcloudExtension()
    ),
    pipe.PolarPipeline(
        handler=pcomp.ArrayHandler(),
        extension=pipe.CurveExtension()
    )
]

# here `data` is treated as some obtained measurements given as
# a numpy.ndarray
pds = [
	pol_pip((data, ["Wind speed", "Wind angle", "Boat speed"]))
	for pol_pip in pol_pips
]
#
for i, pd in enumerate(pds):
   pd.plot_polar(ws=ws, ax=plt.subplot(1, 3, i+1, projection="polar"))

plt.tight_layout()
plt.show()
```
![pipeline_plots](https://user-images.githubusercontent.com/70914876/146170918-66224c66-05c4-49db-a1a5-ddfc2a13b9f1.png)

If we are unhappy with the default behaviour of the pipelines, 
we can customize one or more components of it.


#### Customizing `PolarPipeline`
```python
class MyInfluenceModel(pcomp.InfluenceModel):
    def remove_influence(self, data: dict):
        return np.array([
            data["Wind speed"],
            (data["Wind angle"] + 90)%360,
            data["Boat speed"]**2
        ]).transpose()

    def add_influence(self, pd, influence_data: dict):
        pass


class MyFilter(pcomp.Filter):
    def filter(self, wts):
        return np.logical_or(wts <= 0.2, wts >= 0.8)


def my_model_func(ws, wa, *params):
    return params[0] + params[1]*wa + params[2]*ws + params[3]*ws*wa


my_regressor = pcomp.LeastSquareRegressor(
    model_func=my_model_func,
    init_vals=(1, 2, 3, 4)
)


my_extension = pipe.CurveExtension(
    regressor=my_regressor
)


def my_norm(pt):
    return np.linalg.norm(pt, axis=1)**4


my_pol_pip = pipe.PolarPipeline(
    handler=pcomp.ArrayHandler(),
    im=MyInfluenceModel(),
    weigher=pcomp.CylindricMeanWeigher(radius=2, norm=my_norm),
    extension=my_extension,
    filter_=MyFilter()
)

my_pd = my_pol_pip((data, ["Wind speed", "Wind angle", "Boat speed"]))
```

The customizations above are arbitrary and lead to comparably bad results:

```python
my_pd.plot_polar(ws=ws)
plt.show()
```
![custom_plot](https://user-images.githubusercontent.com/70914876/146348767-f1af3957-8e62-42fa-9f1e-36e872f598c2.png)


#### Including Influences and Weather models
For the next example we initialize a simple influence model and 
a random weather model.

```python
from datetime import timedelta
from datetime import datetime as dt

class MyInfluenceModel(cruise.InfluenceModel):

    def remove_influence(self, data):
        pass

    def add_influence(self, pd, data, **kwargs):
        ws, wa, hdt = data["WS"], data["WA"], data["HDT"]
        twa_spec = (wa - hdt) % 360
        twa = (180 - twa_spec) % 360
        tws = ws
        return pd(tws, twa)


im = MyInfluenceModel()

n, m, k, l = 500, 50, 40, 2

data = 20 * (np.random.random((n, m, k, l)) - 0.5)

wm = cruise.WeatherModel(
    data=data,
    times=[dt.now() + i * timedelta(hours=1) for i in range(n)],
    lats=np.linspace(40, 50, m),
    lons=np.linspace(40, 50, k),
    attrs=["WS", "WA"]
)
```

#### Computing Isochrones
```python
start = (42.5, 43.5)

isochrones = [
    cruise.isochrone(
            pd=pd,
            start=start,
            start_time=dt.now(),
            direction=direction,
            wm=wm,
	    im=im,
            total_time=1 / 3
        )
    for direction in range(0, 360, 5)
]

coordinates, _ = zip(*isochrones)
lats, longs = zip(*coordinates)

for lat, long in coordinates:
    plt.plot([start[0], lat], [start[1], long], color="lightgray")
plt.plot(lats, longs, ls="", marker=".")

plt.show()
```
![icochrone_net](https://user-images.githubusercontent.com/70914876/146554921-befa7bfe-b88f-4c55-93da-8b40aa65f29e.png)


### License 
The `hrosailing` package is published under the 
[Apache 2.0 License](https://choosealicense.com/licenses/apache-2.0/), 
see also [License](LICENSE)


### Citing
[![DOI](https://zenodo.org/badge/409121004.svg)](https://zenodo.org/badge/latestdoi/409121004)

Also see [Citation](CITATION.cff). 


### TODO
[Todo](TODO.md)
