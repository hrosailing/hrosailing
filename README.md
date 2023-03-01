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
![Still in active development. In particular we do not guarantee backwards compatibility to the versions 0.x.x.]!
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

It has the following dependencies:

- `numpy` version 1.22.0
- `scipy` version 1.9.1
- `matplotlib` version 3.4.3
	
For some features it might be necessary to also use:

- `pynmea2` version 1.18.0
- `pandas` version 1.3.3
- `netCDF4` version 1.6.1
- `meteostat` version 1.6.5
	
The `hrosailing` package might also be compatible (in large) with 
other versions of Python, together with others versions of some 
of the used packages. However, this has not been tested properly.


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

# serializes the polar diagram to a .csv file
# in the style of an intern format
pd.to_csv("polar_diagram.csv")
# the default format is the intern format `hro`
pd2 = pol.from_csv("polar_diagram.csv")
```

Currently serialization is only supported for some csv-formats, see also
[csv-format-examples](https://github.com/hrosailing/hrosailing/tree/main/examples/csv-format-examples)
for example files for the currently supported formats. See also 
[Issue #1](https://github.com/hrosailing/hrosailing/issues/1) for a plan
to add more serialization options.


#### Visualizing polar diagrams
```python
import matplotlib.pyplot as plt
import hrosailing.plotting as plot

ws = [10, 20, 30]

plt.subplot(2, 2, 1, projection="hro polar").plot(pd, ws=ws)
plt.subplot(2, 2, 2, projection="hro polar").plot(pd, ws=ws, use_convex_hull=True)
plt.subplot(2, 2, 3, projection="hro flat").plot(pd, ws=ws)
plt.subplot(2, 2, 4, projection="hro color gradient").plot(pd)

plt.show()
```
![flat_plots](https://user-images.githubusercontent.com/70914876/146026223-fc58a914-9b01-47ae-bf9c-6429113dbf4a.png)

3d visualization is also supported.
```python
plot.plot_3d(pd)
plt.show()
```
![output_3d](https://user-images.githubusercontent.com/70914876/220571096-38bf678c-994e-4429-9da6-19dc2d84ffd6.png)



#### Iterate over polar diagram data
We can also directly iterate and/or evaluate the wind angles, 
wind speeds and boat speeds of the polar diagram.

```python
import numpy as np


def random_shifted_pt(pt, mul):
    pt = np.array(pt)
    rand = np.random.random(pt.shape) - 0.5
    rand *= np.array(mul)
    random_pt = pt + rand
    for i in range(3):
        random_pt[i] = max(random_pt[i], 0)
    return random_pt


data = np.array([
    random_shifted_pt([ws, wa, pd(ws, wa)], [10, 5, 2])
    for wa in pd.wind_angles
    for ws in pd.wind_speeds
    for _ in range(6)
])
data = data[np.random.choice(len(data), size=500)]
```


#### Creating polar diagrams from raw data
```python
import hrosailing.pipeline as pipe
import hrosailing.processing as proc

pol_pips = [
    pipe.PolarPipeline(
        data_handler=proc.ArrayHandler(),
        imputator=proc.RemoveOnlyImputator(),
        extension=pipe.TableExtension()
    ),
    pipe.PolarPipeline(
        data_handler=proc.ArrayHandler(),
        imputator=proc.RemoveOnlyImputator(),
        extension=pipe.PointcloudExtension()
    ),
    pipe.PolarPipeline(
        data_handler=proc.ArrayHandler(),
        imputator=proc.RemoveOnlyImputator(),
        extension=pipe.CurveExtension()
    )
]

# here `data` is treated as some obtained measurements given as
# a numpy.ndarray
pds = [
	pol_pip(
        [(data, ["Wind speed", "Wind angle", "Boat speed"])]
    ).polardiagram
	for pol_pip in pol_pips
]
#
for i, pd in enumerate(pds):
   plt.subplot(1, 3, i+1, projection="hro polar").plot(pd, ws=ws)

plt.tight_layout()
plt.show()
```

![pipeline_plots_v2](https://user-images.githubusercontent.com/70914876/220584274-06f31e81-ecee-4825-b911-fef38c26fbef.png)

If we are unhappy with the default behaviour of the pipelines, 
we can customize one or more components of it.


#### Customizing `PolarPipeline`
```python
import hrosailing.models as models

class MyInfluenceModel(models.InfluenceModel):
    def remove_influence(self, data):
        tws = np.array(data["TWS"])
        twa = np.array(data["TWA"])
        bsp = np.array(data["BSP"])
        return np.array([
            tws,
            (twa + 90)%360,
            bsp**2
        ]).transpose()

    def add_influence(self, pd, influence_data: dict):
        pass


class MyFilter(proc.Filter):
    def filter(self, wts):
        return np.logical_or(wts <= 0.2, wts >= 0.8)


def my_model_func(ws, wa, *params):
    return params[0] + params[1]*wa + params[2]*ws + params[3]*ws*wa


my_regressor = proc.LeastSquareRegressor(
    model_func=my_model_func,
    init_vals=(1, 2, 3, 4)
)


my_extension = pipe.CurveExtension(
    regressor=my_regressor
)


def my_norm(pt):
    return np.linalg.norm(pt, axis=1)**4


my_pol_pip = pipe.PolarPipeline(
    data_handler=proc.ArrayHandler(),
    imputator=proc.RemoveOnlyImputator(),
    influence_model=MyInfluenceModel(),
    post_weigher=proc.CylindricMeanWeigher(radius=2, norm=my_norm),
    extension=my_extension,
    post_filter=MyFilter()
)

out = my_pol_pip([(data, ["Wind speed", "Wind angle", "Boat speed"])])
my_pd = out.polardiagram
```

The customizations above are arbitrary and lead to comparably bad results:

```python
plot.plot_polar(my_pd, ws=ws)
plt.show()
```
![custom_plot](https://raw.githubusercontent.com/Loitador41/test_repository/main/.github/images/Figure_Customizing_Pipeline.png)


#### Including Influences and Weather models
For the next example we initialize a simple influence model and 
a random weather model.

```python
from datetime import timedelta
from datetime import datetime as dt


class MyInfluenceModel(models.InfluenceModel):

    def remove_influence(self, data):
        pass

    def add_influence(self, pd, data, **kwargs):
        ws, wa, wave_height = np.array(
            [data["TWS"], data["TWA"], data["WVHGT"]]
        )
        twa = (wa + 5)%360
        tws = ws + ws/wave_height
        return [pd(ws, wa) for ws, wa in zip(tws, twa)]


im = MyInfluenceModel()

n, m, k, l = 500, 50, 40, 3

data = 20 * (np.random.random((n, m, k, l)))

wm = models.GriddedWeatherModel(
    data=data,
    times=[dt.now() + i * timedelta(hours=1) for i in range(n)],
    lats=np.linspace(40, 50, m),
    lons=np.linspace(40, 50, k),
    attrs=["TWS", "TWA", "WVHGT"]
)
```

#### Computing Isochrones
```python
import hrosailing.cruising as cruise

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
