[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linter](https://github.com/hrosailing/hrosailing/actions/workflows/linting.yml/badge.svg)](https://github.com/hrosailing/hrosailing/actions/workflows/linting.yml)
[![tester](https://github.com/hrosailing/hrosailing/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/hrosailing/hrosailing/actions/workflows/build-and-test.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/hrosailing/hrosailing/badge)](https://www.codefactor.io/repository/github/hrosailing/hrosailing)

<p align="center">
    <img src="https://raw.githubusercontent.com/hrosailing/hrosailing/main/logo.png" width=300px height=300px alt="hrosailing">
</p>

hrosailing - Sailing made in Rostock
====================================

You can find the documentation [here](https://hrosailing.github.io/hrosailing/ "hrosailing").

### Compatibility 
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

The `hrosailing` module might also be compatible (in large) with earlier versions of Python (`3.5`, `3.6`), together with some earlier
version of some of the used packages, namely `numpy`, `scipy`, and `matplotlib`, aswell as Python `3.10`, but since it was released 
recently, we can't guarantee that.


### Installation

The recommended way to install `hrosailing` is with 
[pip](http://pypi.python.org/pypi/pip/):
    
    pip install hrosailing

[![PyPI version](https://badge.fury.io/py/hrosailing.svg)](https://badge.fury.io/py/hrosailing)


### License 

The `hrosailing` module is published under the [Apache 2.0 License](https://choosealicense.com/licenses/apache-2.0/), see also [License](LICENSE)

### Examples

In the following we showcase via multiple examples some of the capabilities of the hrosailing module.
To avoid redundancies, all definitions of an example code might be used in the succeeding examples.

For a first example, we download some table with performance diagram data available [here](https://www.seapilot.com/wp-content/uploads/2018/05/60ftmono.txt) and save it as testdata.csv.
This data is given in a tab seperated format which is supported by the keyword ```fmt=array``` of the ```from_csv``` method.
Since this data is only defined for wind angles between 0° and 180° we use the symmetrize function to obtain a symmetric polar diagram with wind angles between 0° and 360°.

```python
import hrosailing.polardiagram as pol
pd = pol.from_csv("testdata.csv", fmt="array").symmetrize()
```

We can use the supported plot functions to get visualizations of the polar diagram.

```python
import matplotlib.pyplot as plt

ws = [10, 20, 30]

pd.plot_polar(ws=ws, ax=plt.subplot(2, 2, 1, projection="polar"))
pd.plot_convex_hull(ws=ws, ax=plt.subplot(2, 2, 2, projection="polar"))
pd.plot_flat(ws=ws, ax=plt.subplot(2, 2, 3))
pd.plot_color_gradient(ax=plt.subplot(2, 2, 4))

plt.show()
```

This results in the following matplotlib diagram:

![flat_plots](https://user-images.githubusercontent.com/70914876/146026223-fc58a914-9b01-47ae-bf9c-6429113dbf4a.png)

We can also visualize the data in a three dimensional plot.

```python
pd.plot_3d()
plt.show()
```

Results in:

![3d_plot](https://user-images.githubusercontent.com/70914876/146153719-826e8c93-09ab-4387-b13c-e942139fcce6.png)

We can also directly iterate and/or evaluate the wind_angles, wind_speeds and boat_speeds of the polar diagram.
For example to artificially create random measurements.

```python
import numpy as np


def random_shifted_pt(pt, mul):
    pt = np.array(pt)
    rand = np.random.random(pt.shape) - 0.5
    rand *= np.array(mul)
    return pt + rand


data = np.array([
    random_shifted_pt([ws, wa, pd.boat_speeds[i, j]], [10, 5, 2])
    for i, ws in enumerate(pd.wind_angles)
    for j, wa in enumerate(pd.wind_speeds)
    for _ in range(6)
])
data = data[np.random.choice(len(data), size=500)]
```

In the following we treat ```data``` like some real life measurement data and try to obtain polar diagrams of different types from it.

```python
import hrosailing.pipeline as pipe
import hrosailing.pipelinecomponents as pcomp


pps = [
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

pds = [pp((data, ["Wind angle", "Wind speed", "Boat speed"])) for pp in pps]

for i, pd in enumerate(pds):
    pd.plot_polar(ws=ws, ax=plt.subplot(1, 3, i+1, projection="polar"))

plt.tight_layout()
plt.show()
```

This results in the following diagram, displaying the different polar diagrams derived from the data:

![pipeline_plots](https://user-images.githubusercontent.com/70914876/146170918-66224c66-05c4-49db-a1a5-ddfc2a13b9f1.png)

If we are unhappy with the default behaviour of the pipelines, we can customize one or more components of it.
In the following example we change many components to showcase the supported modularity:

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


my_pp = pipe.PolarPipeline(
    handler=pcomp.ArrayHandler(),
    im=MyInfluenceModel(),
    weigher=pcomp.CylindricMeanWeigher(radius=2, norm=my_norm),
    extension=my_extension,
    filter_=MyFilter()
)

my_pd = my_pp((data, ["Wind angle", "Wind speed", "Boat speed"]))
```

Of course, the customizations above are arbitrary and lead to comparibly bad results:

![custom_plot](https://user-images.githubusercontent.com/70914876/146348767-f1af3957-8e62-42fa-9f1e-36e872f598c2.png)

It needs more sophisticated approaches for good results.
