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

### Examples


First we import the polardiagram submodule and other useful modules.

```python
>>>import hrosailing.polardiagram as pol
>>>import numpy as np
```

The polardiagram submodule supports three different data types for polar performance diagrams, namely as table, as a pointcloud or as a (three dimensional) curve.
We initialize a table with custom axis resolutions and boat speed data with matching dimensions.

```python
>>>ws_res = [0, 10, 20] # some wind speeds
>>>wa_res = [60,120] # some wind angles
>>>bsps = [[3.95,5.23,5.8],[4.18,5.61,7.1]] # some boat speeds
>>>pd = pol.PolarDiagramTable(ws_res=ws_res, wa_res=wa_res, bsps=bsps) # resulting polar diagram
```

### License 

The `hrosailing` module is published under the [Apache 2.0 License](https://choosealicense.com/licenses/apache-2.0/), see also [License](LICENSE)
