# TODO 

### Documentation
- [ ] Fix search bar on documentation web page
- [ ] Complete examples
- [ ] Make descriptions of parameters and functionalities more verbose
- [ ] Add 'Getting started' section to first site of documentation
- [ ] Add Contributing.md
- [ ] Add links to referenced submodules, classes, exceptions and methods
- [ ] Fix style problems due to conversion from pdoc3 to pdoc

### Tests
- [ ] Testsuite for cruising-Submodule
- [ ] Testsuite for pipeline-Submodule
- [ ] Testsuite for pipelinecomponents-Submodule


### cruising-Submodule
- [ ] Complete implementation of WeatherModel
  - [ ] Find a way to read grib-files in python
  - [ ] Change the current dummy implementation 
  - [ ] Change the default handling for WeatherModel in various functions
- [ ] Complete implementation for helper-functions
  - [ ] wind_direction
- [ ] Make convex_direction and cruising available for interpolated slices
- [ ] Add API for forwarding slice keywords in convex_direction and cruising


### pipelinecomponents-Submodule
- [ ] Implement default InfluenceModel-Subclasses
  - [ ] Change the default handling for InfluenceModel in various functions


### polardiagram-Submodule
- [ ] fix plot_convex_hull() for incomplete polar diagrams


### plotting
- [ ] implement plot_color_gradient()-method for PolarDiagramMultiSails
- [ ] improve the coloring-API for PolarDiagramMultiSails

### new API for plotting
- [ ] create classes inheriting from matplotlib doing the plotting



### Administrative
- [ ] Complete README
- [ ] Complete Module-Docstring for Documentation
