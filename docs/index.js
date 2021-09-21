URLS=[
"hrosailing/index.html",
"hrosailing/polardiagram/index.html",
"hrosailing/pipeline/index.html",
"hrosailing/pipelinecomponents/index.html",
"hrosailing/pipelinecomponents/modelfunctions/index.html",
"hrosailing/pipelinecomponents/regressor.html",
"hrosailing/pipelinecomponents/interpolator.html",
"hrosailing/pipelinecomponents/filter.html",
"hrosailing/pipelinecomponents/influencemodel.html",
"hrosailing/pipelinecomponents/weigher.html",
"hrosailing/pipelinecomponents/neighbourhood.html",
"hrosailing/pipelinecomponents/datahandler.html",
"hrosailing/pipelinecomponents/sampler.html",
"hrosailing/wind.html",
"hrosailing/cruising/index.html"
];
INDEX=[
{
"ref":"hrosailing",
"url":0,
"doc":"The Python package hrosailing provides classes and functions   polar diagrams   sailing   from data  . pipeline  . machine learning  . modular   Installation       hrosailing can be installed using  pip install hrosailing . It requires Python  ., aswell as numpy  ., matplotlib  . and scipy  . to run. Getting Started        - Contributing       License    - Authors    - Valentin Dannenberg (valentin.dannenberg2@uni-rostock.de) Robert Schueler (robert.schueler2@uni-rostock.de)"
},
{
"ref":"hrosailing.polardiagram",
"url":1,
"doc":"PolarDiagram classes to work with and represent PPDs in various forms"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramException",
"url":1,
"doc":"Exception raised if some nonstandard error occurs, while doing something with polar diagrams"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramInitializationException",
"url":1,
"doc":"Exception raised if an error occurs during initialization of a PolarDiagram"
},
{
"ref":"hrosailing.polardiagram.FileReadingException",
"url":1,
"doc":"Exception raised if non-oserror error occurs, when reading a file"
},
{
"ref":"hrosailing.polardiagram.to_csv",
"url":1,
"doc":"See to_csv()-method of PolarDiagram Parameters      csv_path : path-like Path to a .csv-file or where a new .csv file will be created obj : PolarDiagram PolarDiagram instance which will be written to .csv file Raises a FileWritingException if an error occurs whilst writing",
"func":1
},
{
"ref":"hrosailing.polardiagram.from_csv",
"url":1,
"doc":"Reads a .csv file and returns the PolarDiagram instance contained in it Parameters      csv_path : path-like Path to a .csv file fmt : string The \"format\" of the .csv file. - hro: format created by the to_csv-method of the PolarDiagram class - orc: format found at [ORC](https: jieter.github.io/orc-data/site/) - opencpn: format created by [OpenCPN Polar Plugin](https: opencpn.org/OpenCPN/plugins/polar.html) - array tw : bool Specifies if wind data in file should be viewed as true wind Defaults to True Returns    - out : PolarDiagram PolarDiagram instance contained in the .csv file Raises a FileReadingException if an unknown format was specified",
"func":1
},
{
"ref":"hrosailing.polardiagram.pickling",
"url":1,
"doc":"See pickling()-method of PolarDiagram Parameters      pkl_path : path-like Path to a .pkl file or where a new .pkl file will be created obj : PolarDiagram PolarDiagram instance which will be written to .csv file",
"func":1
},
{
"ref":"hrosailing.polardiagram.depickling",
"url":1,
"doc":"Reads a .pkl file and returns the PolarDiagram instance contained in it. Parameters      pkl_path : path-like Path to a .pkl file Returns    - out : PolarDiagram PolarDiagram instance contained in the .pkl file",
"func":1
},
{
"ref":"hrosailing.polardiagram.symmetrize",
"url":1,
"doc":"See symmetrize()-method of PolarDiagram Parameters      obj : PolarDiagram PolarDiagram instance which will be symmetrized Returns    - out : PolarDiagram \"symmetrized\" version of input",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram",
"url":1,
"doc":"Base class for all polar diagrams Abstract Methods         to_csv(csv_path) symmetrize() get_slices(ws) plot_polar( ws, ax=None, colors=(\"green\", \"red\"), show_legend=False, legend_kw=None,  plot_kw ) plot_flat( ws, ax=None, colors=(\"green\", \"red\"), show_legend=False, legend_kw=None,  plot_kw ) plot_3d(ax=None,  plot_kw) plot_color_gradient( ax=None, colors=(\"green\", \"red\"), marker=None, ms=None, show_legend=False,  legend_kw, ) plot_convex_hull( ws, ax=None, colors=(\"green\", \"red\"), show_legend=False, legend_kw=None,  plot_kw, )"
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.pickling",
"url":1,
"doc":"Writes PolarDiagram instance to a .pkl file Parameters      pkl_path: path-like Path to a .pkl file or where a new .pkl file will be created",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.to_csv",
"url":1,
"doc":"This method should, given a path, write a .csv file in the location, containing human readable information about the polar diagram object that called the method",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.symmetrize",
"url":1,
"doc":"This method should return a new PolarDiagram object that is a symmetric (i.e. mirrored along the 0 - 180\u00b0 axis) version of the polar diagram object that called the method",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.get_slices",
"url":1,
"doc":"This method should, given a number of wind speeds, return a list of the given wind speeds as well as wind angles and corresponding boat speeds, that reflect how the vessel behaves at the given wind speeds",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_polar_slice",
"url":1,
"doc":"Creates a polar plot of a given slice of the polar diagram Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_polar()-method of the respective PolarDiagram subclasses ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_polar()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_flat_slice",
"url":1,
"doc":"Creates a cartesian plot of a given slice of the polar diagram Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_flat()-method of the respective PolarDiagram subclass ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_flat()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_polar",
"url":1,
"doc":"This method should create a polar plot of one or more slices, corresponding to  ws , of the polar diagram object.",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_flat",
"url":1,
"doc":"This method should create a cartesian plot of one or more slices, corresponding to  ws , of the polar diagram object",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_3d",
"url":1,
"doc":"This method should create a 3d plot of the polar diagram object",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_color_gradient",
"url":1,
"doc":"This method should create 'wind speed vs. wind angle' color gradient plot of the polar diagram object with respect to the corresponding boat speeds",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_convex_hull_slice",
"url":1,
"doc":"Computes the convex hull of a given slice of the polar diagram and creates a polar plot of it Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_convex_hull()-method of the respective PolarDiagram subclass ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_convex_hull()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_convex_hull",
"url":1,
"doc":"This method should compute the convex hull of one or multiple slices, corresponding to  ws , of the polar diagram and then create a polar plot of them",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable",
"url":1,
"doc":"A class to represent, visualize and work with a polar diagram in the form of a table. Parameters      ws_res : array_like or positive int/float, optional Wind speeds that will correspond to the columns of the table Can either be a sequence of length cdim or an int/float value If a number num is passed, numpy.arange(num, 40, num) will be assigned to ws_res If nothing is passed, it will default to numpy.arange(2, 42, 2) wa_res : array_like or positive int/float, optional Wind angles that will correspond to the rows of the table. Should be between 0\u00b0 and 360\u00b0 Can either be sequence of length rdim or an int/float value If a number num is passed, numpy.arange(num, 360, num) will be assigned to wa_res If nothing is passed, it will default to numpy.arange(0, 360, 5) bsps : array_like, optional Boatspeeds that will correspond to the entries of the table Should be broadcastable to the shape (rdim, cdim) If nothing is passed it will default to numpy.zeros rdim, cdim Raises a PolarDiagramException - if bsps is not array_like or of a fitting shape Examples     >>> pd = PolarDiagramTable() >>> pd.wind_speeds [ 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40] >>> pd.wind_angles [ 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 250 255 260 265 270 275 280 285 290 295 300 305 310 315 320 325 330 335 340 345 350 355] >>> pd = PolarDiagramTable(ws_res = [6, 8, 10, 12, 14],  . wa_res = [52, 60, 75, 90, 110, 120, 135]) >>> print(pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 0.00 0.00 0.00 0.00 0.00 60.0 0.00 0.00 0.00 0.00 0.00 75.0 0.00 0.00 0.00 0.00 0.00 90.0 0.00 0.00 0.00 0.00 0.00 110.0 0.00 0.00 0.00 0.00 0.00 120.0 0.00 0.00 0.00 0.00 0.00 135.0 0.00 0.00 0.00 0.00 0.00 >>> pd = PolarDiagramTable(  . bsps=[  . [5.33, 6.32, 6.96, 7.24, 7.35],  . [5.64, 6.61, 7.14, 7.42, 7.56],  . [5.89, 6.82, 7.28, 7.59, 7.84],  . [5.92, 6.98, 7.42, 7.62, 7.93],  . [5.98, 7.07, 7.59, 8.02, 8.34],  . [5.8, 6.95, 7.51, 7.98, 8.52],  . [5.2, 6.41, 7.19, 7.66, 8.14]  . ],  . ws_res=[6, 8, 10, 12, 14],  . wa_res=[52, 60, 75, 90, 110, 120, 135],  . ) >>> print(pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 5.33 6.32 6.96 7.24 7.35 60.0 5.64 6.61 7.14 7.42 7.56 75.0 5.89 6.82 7.28 7.59 7.84 90.0 5.92 6.98 7.42 7.62 7.93 110.0 5.98 7.07 7.59 8.02 8.34 120.0 5.80 6.95 7.51 7.98 8.52 135.0 5.20 6.41 7.19 7.66 8.14"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.__getitem__",
"url":1,
"doc":"Parameters      key : Returns    - bsp :",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.wind_angles",
"url":1,
"doc":"Returns a read only version of self._res_wind_angle"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.wind_speeds",
"url":1,
"doc":"Returns a read only version of self._res_wind_speed"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.boat_speeds",
"url":1,
"doc":"Returns a read only version of self._boat_speeds"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.to_csv",
"url":1,
"doc":"Creates a .csv file with delimiter ',' and the following format: PolarDiagramTable Wind speed resolution: self.wind_speeds Wind angle resolution: self.wind_angles Boat speeds: self.boat_speeds Parameters      csv_path : path-like Path to a .csv file or where a new .csv file will be created fmt : string Raises a FileWritingException if an unknown format was specified Examples     >>> pd = PolarDiagramTable(  . bsps=[  . [5.33, 6.32, 6.96, 7.24, 7.35],  . [5.64, 6.61, 7.14, 7.42, 7.56],  . [5.89, 6.82, 7.28, 7.59, 7.84],  . [5.92, 6.98, 7.42, 7.62, 7.93],  . [5.98, 7.07, 7.59, 8.02, 8.34],  . [5.8, 6.95, 7.51, 7.98, 8.52],  . [5.2, 6.41, 7.19, 7.66, 8.14]  . ],  . ws_res=[6, 8, 10, 12, 14],  . wa_res=[52, 60, 75, 90, 110, 120, 135],  . ) >>> print(pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 5.33 6.32 6.96 7.24 7.35 60.0 5.64 6.61 7.14 7.42 7.56 75.0 5.89 6.82 7.28 7.59 7.84 90.0 5.92 6.98 7.42 7.62 7.93 110.0 5.98 7.07 7.59 8.02 8.34 120.0 5.80 6.95 7.51 7.98 8.52 135.0 5.20 6.41 7.19 7.66 8.14 >>> pd.to_csv(\"example.csv\") >>> pd2 = from_csv(\"example.csv\") >>> print(pd2) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 5.33 6.32 6.96 7.24 7.35 60.0 5.64 6.61 7.14 7.42 7.56 75.0 5.89 6.82 7.28 7.59 7.84 90.0 5.92 6.98 7.42 7.62 7.93 110.0 5.98 7.07 7.59 8.02 8.34 120.0 5.80 6.95 7.51 7.98 8.52 135.0 5.20 6.41 7.19 7.66 8.14",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.symmetrize",
"url":1,
"doc":"Constructs a symmetric version of the polar diagram, by mirroring it at the 0\u00b0 - 180\u00b0 axis and returning a new instance Warning    - Should only be used if all the wind angles of the initial polar diagram are on one side of the 0\u00b0 - 180\u00b0 axis, otherwise this can lead to duplicate data, which can overwrite or live alongside old data Examples     >>> pd = PolarDiagramTable(  . bsps=[  . [5.33, 6.32, 6.96, 7.24, 7.35],  . [5.64, 6.61, 7.14, 7.42, 7.56],  . [5.89, 6.82, 7.28, 7.59, 7.84],  . [5.92, 6.98, 7.42, 7.62, 7.93],  . ],  . ws_res = [6, 8, 10, 12, 14],  . wa_res = [52, 60, 75, 90]  . ) >>> print(pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 5.33 6.32 6.96 7.24 7.35 60.0 5.64 6.61 7.14 7.42 7.56 75.0 5.89 6.82 7.28 7.59 7.84 90.0 5.92 6.98 7.42 7.62 7.93 >>> sym_pd = pd.symmetrize() >>> print(sym_pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 5.33 6.32 6.96 7.24 7.35 60.0 5.64 6.61 7.14 7.42 7.56 75.0 5.89 6.82 7.28 7.59 7.84 90.0 5.92 6.98 7.42 7.62 7.93 270.0 5.92 6.98 7.42 7.62 7.93 285.0 5.89 6.82 7.28 7.59 7.84 300.0 5.64 6.61 7.14 7.42 7.56 308.0 5.33 6.32 6.96 7.24 7.35",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.change_entries",
"url":1,
"doc":"Changes specified entries in the table Parameters      new_bsps: array_like of matching shape Sequence containing the new boat speeds to be inserted in the specified entries ws: Iterable or int or float, optional Element(s) of self.wind_speeds, specifying the columns, where new boat speeds will be inserted If nothing is passed it will default to self.wind_speeds wa: Iterable or int or float, optional Element(s) of self.wind_angles, specifiying the rows, where new boat speeds will be inserted If nothing is passed it will default to self.wind_angles Examples     >>> pd = PolarDiagramTable(  . ws_res = [6, 8, 10, 12, 14],  . wa_res = [52, 60, 75, 90, 110, 120, 135]  . ) >>> print(pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 0.00 0.00 0.00 0.00 0.00 60.0 0.00 0.00 0.00 0.00 0.00 75.0 0.00 0.00 0.00 0.00 0.00 90.0 0.00 0.00 0.00 0.00 0.00 110.0 0.00 0.00 0.00 0.00 0.00 120.0 0.00 0.00 0.00 0.00 0.00 135.0 0.00 0.00 0.00 0.00 0.00 >>> pd.change_entries(  . new_bsps=[5.33, 5.64, 5.89, 5.92, 5.98, 5.8, 5.2],  . ws=6  . ) >>> print(pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 5.33 0.00 0.00 0.00 0.00 60.0 5.64 0.00 0.00 0.00 0.00 75.0 5.89 0.00 0.00 0.00 0.00 90.0 5.92 0.00 0.00 0.00 0.00 110.0 5.98 0.00 0.00 0.00 0.00 120.0 5.80 0.00 0.00 0.00 0.00 135.0 5.20 0.00 0.00 0.00 0.00 >>> pd.change_entries(  . new_bsps=[5.7, 6.32, 6.96, 7.24, 7.35],  . wa=52  . ) >>> print(pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 5.70 6.32 6.96 7.24 7.35 60.0 5.64 0.00 0.00 0.00 0.00 75.0 5.89 0.00 0.00 0.00 0.00 90.0 5.92 0.00 0.00 0.00 0.00 110.0 5.98 0.00 0.00 0.00 0.00 120.0 5.80 0.00 0.00 0.00 0.00 135.0 5.20 0.00 0.00 0.00 0.00 >>> pd.change_entries(new_bsps=[5.33], ws=6, wa=52) >>> print(pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 5.33 6.32 6.96 7.24 7.35 60.0 5.64 0.00 0.00 0.00 0.00 75.0 5.89 0.00 0.00 0.00 0.00 90.0 5.92 0.00 0.00 0.00 0.00 110.0 5.98 0.00 0.00 0.00 0.00 120.0 5.80 0.00 0.00 0.00 0.00 135.0 5.20 0.00 0.00 0.00 0.00 >>> pd.change_entries(  . new_bsps= 6.61, 7.14], [6.82, 7.28 ,  . ws=[8, 10],  . wa=[60, 75]  . ) >>> print(pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 5.33 6.32 6.96 7.24 7.35 60.0 5.64 6.61 7.14 0.00 0.00 75.0 5.89 6.82 7.28 0.00 0.00 90.0 5.92 0.00 0.00 0.00 0.00 110.0 5.98 0.00 0.00 0.00 0.00 120.0 5.80 0.00 0.00 0.00 0.00 135.0 5.20 0.00 0.00 0.00 0.00",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.get_slices",
"url":1,
"doc":"For given wind speeds, return the slices of the polar diagram corresponding to them The slices are equal to the corresponding columns of the table together with self.wind_angles Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram table, given as either - a tuple of length 2 specifying an interval of considered wind speeds - an iterable containing only elements of self.wind_speeds - a single element of self.wind_speeds If nothing it passed, it will default to self.wind_speeds Returns    - slices : tuple Slices of the polar diagram, given as a tuple of length 3, consisting of the given wind speeds  ws , self.wind_angles (in rad) and an array with the corresponding columns of the table",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_polar",
"url":1,
"doc":"Creates a polar plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram table, given as either - a tuple of length 2 specifying an interval of considered wind speeds - an iterable containing only elements of self.wind_speeds - a single element of self.wind_speeds The slices are then equal to the corresponding columns of the table together with self.wind_angles If nothing it passed, it will default to self.wind_speeds ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException - if at least one element of ws is not in self.wind_speeds - the given interval doesn't contain any slices of the polar diagram Examples     >>> import matplotlib.pyplot as pyplot >>> pd = from_csv(\"src/polar_diagrams/orc/A-35.csv\", fmt=\"orc\") >>> pd.plot_polar(  . ws=[6, 8], show_legend=True, ls=\"-\", lw=1.5, marker=  . ) >>> pyplot.show()  image /examples/table_plot_polar.png Can also be combined with symmetrize() like so to achieve a symmetric plot >>> pd.symmetrize().plot_polar(  . ws=[6, 8], show_legend=True, ls=\"-\", lw=1.5, marker=  . ) >>> pyplot.show()  image /examples/sym_table_plot_polar.png",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_flat",
"url":1,
"doc":"Creates a cartesian plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram table, given as either - a tuple of length 2 specifying an interval of considered wind speeds - an iterable containing only elements of self.wind_speeds - a single element of self.wind_speeds The slices are then equal to the corresponding columns of the table together with self.wind_angles If nothing it passed, it will default to self.wind_speeds ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException - if at least one element of ws is not in self.wind_speeds - the given interval doesn't contain any slices of the polar diagram Examples     >>> import matplotlib.pyplot as pyplot >>> pd = from_csv(\"src/polar_diagrams/orc/A-35.csv\", fmt=\"orc\") >>> pd.plot_flat(  . ws=[6, 8], show_legend=True, ls=\"-\", lw=1.5, marker=  . ) >>> pyplot.show()  image /examples/table_plot_flat.png Can also be combined with symmetrize() like so to achieve a symmetric plot >>> pd.symmetrize().plot_flat(  . ws=[6, 8], show_legend=True, ls=\"-\", lw=1.5, marker=  . ) >>> pyplot.show()  image /examples/sym_table_plot_flat.png",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_3d",
"url":1,
"doc":"Creates a 3d plot of the polar diagram Parameters      ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_color_gradient",
"url":1,
"doc":"Creates a 'wind speed vs. wind angle' color gradient plot of the polar diagram with respect to the corresponding boat speeds Parameters      ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple of length 2, optional Colors which specify the color gradient with which the polar diagram will be plotted. Defaults to (\"green\", \"red\") marker : matplotlib.markers.Markerstyle or equivalent, optional Markerstyle for the created scatter plot If nothing is passed, it will default to \"o\" ms : float or array_like of fitting shape, optional Marker size in points 2 If nothing is passed, it will use the default of the matplotlib.pyplot.scatter function show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot Legend will be a matplotlib.colorbar.Colorbar object. Defaults to False legend_kw : Keyword arguments Keyword arguments to be passed to the matplotlib.colorbar.Colorbar class to change position and appearence of the legend. Will only be used if show_legend is True If nothing is passed, it will default to {}",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_convex_hull",
"url":1,
"doc":"Computes the (seperate) convex hull of one or more slices of the polar diagram and creates a polar plot of them Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram table, given as either - a tuple of length 2 specifying an interval of considered wind speeds - an iterable containing only elements of self.wind_speeds - a single element of self.wind_speeds The slices are then equal to the corresponding columns of the table together with self.wind_angles If nothing it passed, it will default to self.wind_speeds ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException - if at least one element of ws is not in self.wind_speeds - the given interval doesn't contain any slices of the polar diagram",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.pickling",
"url":1,
"doc":"Writes PolarDiagram instance to a .pkl file Parameters      pkl_path: path-like Path to a .pkl file or where a new .pkl file will be created",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_polar_slice",
"url":1,
"doc":"Creates a polar plot of a given slice of the polar diagram Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_polar()-method of the respective PolarDiagram subclasses ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_polar()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_flat_slice",
"url":1,
"doc":"Creates a cartesian plot of a given slice of the polar diagram Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_flat()-method of the respective PolarDiagram subclass ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_flat()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_convex_hull_slice",
"url":1,
"doc":"Computes the convex hull of a given slice of the polar diagram and creates a polar plot of it Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_convex_hull()-method of the respective PolarDiagram subclass ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_convex_hull()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails",
"url":1,
"doc":"A class to represent, visualize and work with a polar diagram made up of multiple sets of sails, represented by a PolarDiagramTable Parameters      pds : Iterable of PolarDiagramTable objects sails : Iterable, optional Raises a PolarDiagramException if"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.sails",
"url":1,
"doc":""
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.wind_speeds",
"url":1,
"doc":""
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.tables",
"url":1,
"doc":""
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.__getitem__",
"url":1,
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.to_csv",
"url":1,
"doc":"Parameters      csv_path : path_like Path to a .csv file or where a new .csv file will be created",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.symmetrize",
"url":1,
"doc":"Constructs a symmetric version of the polar diagram, by mirroring each PolarDiagramTable at the 0\u00b0 - 180\u00b0 axis and returning a new instance. See also the symmetrize()-method of the PolarDiagramTable class Warning    - Should only be used if all the wind angles of the PolarDiagramTables are each on one side of the 0\u00b0 - 180\u00b0 axis, otherwise this can lead to duplicate data, which can overwrite or live alongside old data",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.get_slices",
"url":1,
"doc":"For given wind speeds, return the slices of the polar diagram corresponding to them The slices are equal to the corresponding columns of the table together with self.wind_angles Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram table, given as either - a tuple of length 2 specifying an interval of considered wind speeds - an iterable containing only elements of self.wind_speeds - a single element of self.wind_speeds If nothing it passed, it will default to self.wind_speeds Returns    - slices : tuple",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.plot_polar",
"url":1,
"doc":"Creates a polar plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram table, given as either - a tuple of length 2 specifying an interval of considered wind speeds - an iterable containing only elements of self.wind_speeds - a single element of self.wind_speeds The slices are then equal to the corresponding columns of the table together with self.wind_angles If nothing it passed, it will default to self.wind_speeds ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException - if at least one element of ws is not in self.wind_speeds - the given interval doesn't contain any slices of the polar diagram",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.plot_flat",
"url":1,
"doc":"Creates a cartesian plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram table, given as either - a tuple of length 2 specifying an interval of considered wind speeds - an iterable containing only elements of self.wind_speeds - a single element of self.wind_speeds The slices are then equal to the corresponding columns of the table together with self.wind_angles If nothing it passed, it will default to self.wind_speeds ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException - if at least one element of ws is not in self.wind_speeds - the given interval doesn't contain any slices of the polar diagram",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.plot_3d",
"url":1,
"doc":"Creates a 3d plot of the polar diagram Parameters      ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.plot_color_gradient",
"url":1,
"doc":"Parameters      ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple of length 2, optional Colors which specify the color gradient with which the polar diagram will be plotted. Defaults to (\"green\", \"red\") marker : matplotlib.markers.Markerstyle or equivalent, optional Markerstyle for the created scatter plot If nothing is passed, it will default to \"o\" ms : float or array_like of fitting shape, optional Marker size in points 2 If nothing is passed, it will use the default of the matplotlib.pyplot.scatter function show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot Legend will be a matplotlib.colorbar.Colorbar object. Defaults to False legend_kw : Keyword arguments Keyword arguments to be passed to the matplotlib.colorbar.Colorbar class to change position and appearence of the legend. Will only be used if show_legend is True If nothing is passed, it will default to {}",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.plot_convex_hull",
"url":1,
"doc":"Computes the (seperate) convex hull of one or more slices of the polar diagram and creates a polar plot of them Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram table, given as either - a tuple of length 2 specifying an interval of considered wind speeds - an iterable containing only elements of self.wind_speeds - a single element of self.wind_speeds The slices are then equal to the corresponding columns of the table together with self.wind_angles If nothing it passed, it will default to self.wind_speeds ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException if at least one element of ws is not in self.wind_speeds",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.pickling",
"url":1,
"doc":"Writes PolarDiagram instance to a .pkl file Parameters      pkl_path: path-like Path to a .pkl file or where a new .pkl file will be created",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.plot_polar_slice",
"url":1,
"doc":"Creates a polar plot of a given slice of the polar diagram Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_polar()-method of the respective PolarDiagram subclasses ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_polar()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.plot_flat_slice",
"url":1,
"doc":"Creates a cartesian plot of a given slice of the polar diagram Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_flat()-method of the respective PolarDiagram subclass ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_flat()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.plot_convex_hull_slice",
"url":1,
"doc":"Computes the convex hull of a given slice of the polar diagram and creates a polar plot of it Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_convex_hull()-method of the respective PolarDiagram subclass ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_convex_hull()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve",
"url":1,
"doc":"A class to represent, visualize and work with a polar diagram given by a fitted curve/surface. Parameters      f : function Curve/surface that describes the polar diagram, given as a function, with the signature f(ws, wa,  params) -> bsp, where ws and wa should be array_like of shape (n,). should then also be an array_like of shape (n,) params : tuple or Sequence Optimal parameters for f radians : bool, optional Specifies if f takes the wind angles to be in radians or degrees Defaults to False"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.__call__",
"url":1,
"doc":"Call self as a function.",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.curve",
"url":1,
"doc":"Returns a read only version of self._f"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.parameters",
"url":1,
"doc":"Returns a read only version of self._params"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.radians",
"url":1,
"doc":"Returns a read only version of self._rad"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.to_csv",
"url":1,
"doc":"Creates a .csv file with delimiter ':' and the following format: PolarDiagramCurve Function: self.curve.__name__ Radians: self.rad Parameters: self.parameters Parameters      csv_path : path-like Path to a .csv file or where a new .csv file will be created Raises a FileWritingException if an error occurs whilst writing",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.symmetrize",
"url":1,
"doc":"Constructs a symmetric version of the polar diagram, by mirroring it at the 0\u00b0 - 180\u00b0 axis and returning a new instance",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.get_slices",
"url":1,
"doc":"For given wind speeds, return the slices of the polar diagram corresponding to them Slices are equal to self(w, wa) where w goes through the given values in  ws and wa goes through a fixed number of angles between 0\u00b0 and 360\u00b0 Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram given as either - a tuple of length 2, specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of specific wind speeds - a single wind speed If nothing is passed, it will default to (0, 20) stepsize : positive int or float, optional Specfies the amount of slices taken from the given wind speed interval Will only be used if  ws is a tuple of length 2 If nothing is passed, it will default to ws[1] - ws[0] Returns    - slices : tuple Slices of the polar diagram, given as a tuple of length 3, consisting of the given wind speeds  ws , self.wind_angles (in rad) and a list of arrays containing the corresponding boat speeds",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_polar",
"url":1,
"doc":"Creates a polar plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram given as either - a tuple of length 2, specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of specific wind speeds - a single wind speed Slices will then equal self(w, wa) where w goes through the given values in  ws and wa goes through a fixed number of angles between 0\u00b0 and 360\u00b0 If nothing is passed, it will default to (0, 20) stepsize : positive int or float, optional Specfies the amount of slices taken from the given wind speed interval Will only be used if  ws is a tuple of length 2 If nothing is passed, it will default to ws[1] - ws[0] ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_flat",
"url":1,
"doc":"Creates a cartesian plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram given as either - a tuple of length 2, specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of specific wind speeds - a single wind speed Slices will then equal self(w, wa) where w goes through the given values in  ws and wa goes through a fixed number of angles between 0\u00b0 and 360\u00b0 If nothing is passed, it will default to (0, 20) stepsize : positive int or float, optional Specfies the amount of slices taken from the given wind speed interval Will only be used if  ws is a tuple of length 2 If nothing is passed, it will default to ws[1] - ws[0] ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_3d",
"url":1,
"doc":"Creates a 3d plot of a part of the polar diagram Parameters      ws : tuple of length 2, optional A region of the polar diagram given as an interval of wind speeds Slices will then equal self(w, wa) where w goes through the given values in  ws and wa goes through a fixed number of angles between 0\u00b0 and 360\u00b0 If nothing is passed, it will default to (0, 20) stepsize : positive int or float, optional Specfies the amount of slices taken from the given interval in  ws If nothing is passed, it will default to 100 ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors :",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_color_gradient",
"url":1,
"doc":"Creates a 'wind speed vs. wind angle' color gradient plot of a part of the polar diagram with respect to the corresponding boat speeds Parameters      ws : tuple of length 3, optional A region of the polar diagram given as an interval of wind speeds Slices will then equal self(w, wa) where w goes through the given values in  ws and wa goes through a fixed number of angles between 0\u00b0 and 360\u00b0 If nothing is passed, it will default to (0, 20) stepsize : positive int or float, optional Specfies the amount of slices taken from the given interval in  ws If nothing is passed, it will default to 100 ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple of length 2, optional Colors which specify the color gradient with which the polar diagram will be plotted. Defaults to (\"green\", \"red\") marker : matplotlib.markers.Markerstyle or equivalent, optional Markerstyle for the created scatter plot If nothing is passed, it will default to \"o\" ms : float or array_like of fitting shape, optional Marker size in points 2 If nothing is passed, it will use the default of the matplotlib.pyplot.scatter function show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot Legend will be a matplotlib.colorbar.Colorbar object. Defaults to False legend_kw : Keyword arguments Keyword arguments to be passed to the matplotlib.colorbar.Colorbar class to change position and appearence of the legend. Will only be used if show_legend is True If nothing is passed, it will default to {}",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_convex_hull",
"url":1,
"doc":"Computes the (seperate) convex hull of one or more slices of the polar diagram and creates a polar plot of them Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram given as either - a tuple of length 2, specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of specific wind speeds - a single wind speed Slices will then equal self(w, wa) where w goes through the given values in  ws and wa goes through a fixed number of angles between 0\u00b0 and 360\u00b0 If nothing is passed, it will default to (0, 20) stepsize : positive int or float, optional Specfies the amount of slices taken from the given wind speed interval Will only be used if  ws is a tuple of length 2 If nothing is passed, it will default to ws[1] - ws[0] ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.pickling",
"url":1,
"doc":"Writes PolarDiagram instance to a .pkl file Parameters      pkl_path: path-like Path to a .pkl file or where a new .pkl file will be created",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_polar_slice",
"url":1,
"doc":"Creates a polar plot of a given slice of the polar diagram Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_polar()-method of the respective PolarDiagram subclasses ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_polar()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_flat_slice",
"url":1,
"doc":"Creates a cartesian plot of a given slice of the polar diagram Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_flat()-method of the respective PolarDiagram subclass ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_flat()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_convex_hull_slice",
"url":1,
"doc":"Computes the convex hull of a given slice of the polar diagram and creates a polar plot of it Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_convex_hull()-method of the respective PolarDiagram subclass ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_convex_hull()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud",
"url":1,
"doc":"A class to represent, visualize and work with a polar diagram given by a point cloud Parameters      pts : array_like of shape (n, 3), optional Initial points of the point cloud, given as a sequence of points consisting of wind speed, wind angle and boat speed If nothing is passed, the point cloud will be initialized as an empty point cloud tw : bool, optional Specifies if the given wind data should be viewed as true wind If False, wind data will be converted to true wind Defaults to True"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.wind_speeds",
"url":1,
"doc":"Returns all unique wind speeds in the point cloud"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.wind_angles",
"url":1,
"doc":"Returns all unique wind angles in the point cloud"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.boat_speeds",
"url":1,
"doc":"Returns all occuring boat speeds in the point cloud (including duplicates)"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.points",
"url":1,
"doc":"Returns a read only version of self._pts"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.to_csv",
"url":1,
"doc":"Creates a .csv file with delimiter ',' and the following format: PolarDiagramPointcloud True wind speed ,True wind angle ,Boat speed self.points Parameters      csv_path : path-like Path to a .csv-file or where a new .csv file will be created",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.symmetrize",
"url":1,
"doc":"Constructs a symmetric version of the polar diagram, by mirroring it at the 0\u00b0 - 180\u00b0 axis and returning a new instance Warning    - Should only be used if all the wind angles of the initial polar diagram are on one side of the 0\u00b0 - 180\u00b0 axis, otherwise this can result in the construction of duplicate points, that might overwrite or live alongside old points",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.add_points",
"url":1,
"doc":"Adds additional points to the point cloud Parameters      new_pts: array_like of shape (n, 3) New points to be added to the point cloud given as a sequence of points consisting of wind speed, wind angle and boat speed tw : bool, optional Specifies if the given wind data should be viewed as true wind If False, wind data will be converted to true wind Defaults to True",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.get_slices",
"url":1,
"doc":"For given wind speeds, return the slices of the polar diagram corresponding to them A slice then consists of all rows in self.wind_speeds whose first entry lies in the interval given by w in  ws Parameters      ws : tuple of length 2, iterable , int or float, optional Slices of the polar diagram given as either - a tuple of length 2 specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of tuples of length 2 and int/float values which will be interpreted as individual slices. If a w in  ws is an int or float, the given interval will be determined by the parameter  range_ . If it is a tuple, it will be interpreted as an inverval as is - a single wind speed. The given interval is then determined by the parameter  range_ If nothing is passed, it will default to (min(self.wind_speeds), max(self.wind_speeds stepsize : positive int, optional Specfies the amount of slices taken from the given interval in  ws Will only be used if  ws is a tuple of length 2 If nothing is passed it will default to int(round(ws[1] - ws[0] range_ : positive int or float, optional Used to convert and int or float w in  ws to the interval (w - range_, w + range_ Will only be used if  ws is int or float or if any w in  ws is an int or float Defaults to 1 Returns    - slices : tuple",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_polar",
"url":1,
"doc":"Creates a polar plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable , int or float, optional Slices of the polar diagram given as either - a tuple of length 2 specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of tuples of length 2 and int/float values which will be interpreted as individual slices. If a w in  ws is an int or float, the given interval will be determined by the parameter  range_ . If it is a tuple, it will be interpreted as an inverval as is - a single wind speed. The given interval is then determined by the parameter  range_ A slice then consists of all rows in self.wind_speeds whose first entry lies in the interval given by w in  ws If nothing is passed, it will default to (min(self.wind_speeds), max(self.wind_speeds stepsize : positive int, optional Specfies the amount of slices taken from the given interval in  ws Will only be used if  ws is a tuple of length 2 If nothing is passed it will default to int(round(ws[1] - ws[0] range_ : positive int or float, optional Used to convert and int or float w in  ws to the interval (w - range_, w + range_ Will only be used if  ws is int or float or if any w in  ws is an int or float Defaults to 1 ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException if ws is given as a single value or a list and there is a value w in ws, such that there are no rows in self.points whose first entry is equal to w",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_flat",
"url":1,
"doc":"Creates a cartesian plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable , int or float, optional Slices of the polar diagram given as either - a tuple of length 2 specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of tuples of length 2 and int/float values which will be interpreted as individual slices. If a w in  ws is an int or float, the given interval will be determined by the parameter  range_ . If it is a tuple, it will be interpreted as an inverval as is - a single wind speed. The given interval is then determined by the parameter  range_ A slice then consists of all rows in self.wind_speeds whose first entry lies in the interval given by w in  ws If nothing is passed, it will default to (min(self.wind_speeds), max(self.wind_speeds) stepsize : positive int, optional Specfies the amount of slices taken from the given interval in  ws Will only be used if  ws is a tuple of length 2 If nothing is passed it will default to int(round(ws[1] - ws[0] range_ : positive int or float, optional Used to convert and int or float w in  ws to the interval (w - range_, w + range_ Will only be used if  ws is int or float or if any w in  ws is an int or float Defaults to 1 ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException if ws is given as a single value or a list and there is a value w in ws, such that there are no rows in self.points whose first entry is equal to w",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_3d",
"url":1,
"doc":"Creates a 3d plot of the polar diagram Parameters      ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException if there are no points in the point cloud",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_color_gradient",
"url":1,
"doc":"Creates a 'wind speed vs. wind angle' color gradient plot of the polar diagram with respect to the corresponding boat speeds Parameters      ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple of length 2, optional Colors which specify the color gradient with which the polar diagram will be plotted. Defaults to (\"green\", \"red\") marker : matplotlib.markers.Markerstyle or equivalent, optional Markerstyle for the created scatter plot If nothing is passed, it will default to \"o\" ms : float or array_like of fitting shape, optional Marker size in points 2 If nothing is passed, it will use the default of the matplotlib.pyplot.scatter function show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot Legend will be a matplotlib.colorbar.Colorbar object. Defaults to False legend_kw : Keyword arguments Keyword arguments to be passed to the matplotlib.colorbar.Colorbar class to change position and appearence of the legend. Will only be used if show_legend is True If nothing is passed, it will default to {} Raises a PolarDiagramException if there are no points in the point cloud",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_convex_hull",
"url":1,
"doc":"Computes the (seperate) convex hull of one or more slices of the polar diagram and creates a polar plot of them Parameters      ws : tuple of length 2, iterable , int or float, optional Slices of the polar diagram given as either - a tuple of length 2 specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of tuples of length 2 and int/float values which will be interpreted as individual slices. If a w in  ws is an int or float, the given interval will be determined by the parameter  range_ . If it is a tuple, it will be interpreted as an inverval as is - a single wind speed. The given interval is then determined by the parameter  range_ A slice then consists of all rows in self.wind_speeds whose first entry lies in the interval given by w in  ws If nothing is passed, it will default to (min(self.wind_speeds), max(self.wind_speeds) stepsize : positive int, optional Specfies the amount of slices taken from the given interval in  ws Will only be used if  ws is a tuple of length 2 Defaults to int(round(ws[1] - ws[0] range_ : positive int or float, optional Will only be used if  ws is int or float or if any w in  ws is an int or float Defaults to 1 ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException if ws is given as a single value or a list and there is a value w in ws, such that there are no rows in self.points whose first entry is equal to w",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.pickling",
"url":1,
"doc":"Writes PolarDiagram instance to a .pkl file Parameters      pkl_path: path-like Path to a .pkl file or where a new .pkl file will be created",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_polar_slice",
"url":1,
"doc":"Creates a polar plot of a given slice of the polar diagram Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_polar()-method of the respective PolarDiagram subclasses ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_polar()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_flat_slice",
"url":1,
"doc":"Creates a cartesian plot of a given slice of the polar diagram Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_flat()-method of the respective PolarDiagram subclass ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_flat()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_convex_hull_slice",
"url":1,
"doc":"Computes the convex hull of a given slice of the polar diagram and creates a polar plot of it Parameters      ws : int/float Slice of the polar diagram For a description of what the slice is made of, see the plot_convex_hull()-method of the respective PolarDiagram subclass ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException, if the plot_convex_hull()-method of the respective PolarDiagram subclass raises one",
"func":1
},
{
"ref":"hrosailing.pipeline",
"url":2,
"doc":"Pipeline to create PPDs from raw data"
},
{
"ref":"hrosailing.pipeline.PipelineException",
"url":2,
"doc":"Exception raised if an error occurs in the pipeline"
},
{
"ref":"hrosailing.pipeline.PipelineExtension",
"url":2,
"doc":"Base class for all pipeline extensions Abstract Methods         process(w_pts)"
},
{
"ref":"hrosailing.pipeline.PipelineExtension.process",
"url":2,
"doc":"This method, given an instance of WeightedPoints, should return a polar diagram object, which represents the trends and data contained in the WeightedPoints instance",
"func":1
},
{
"ref":"hrosailing.pipeline.PolarPipeline",
"url":2,
"doc":"A Pipeline class to create polar diagrams from raw data Parameters      extension: PipelineExtension handler : DataHandler weigher : Weigher, optional filter_ : Filter, optional"
},
{
"ref":"hrosailing.pipeline.TableExtension",
"url":2,
"doc":"Parameters      w_res : , optional neighbourhood : Neighbourhood, optional Defaults to Ball() interpolator : Interpolator, optional Defaults to ArithmeticMeanInterpolator(50)"
},
{
"ref":"hrosailing.pipeline.TableExtension.process",
"url":2,
"doc":"Parameters      w_pts : WeightedPoints Returns    - pd : PolarDiagramTable",
"func":1
},
{
"ref":"hrosailing.pipeline.CurveExtension",
"url":2,
"doc":"Parameters      regressor : Regressor, optional Defaults to  ODRegressor( model_func=ws_s_s_dt_wa_gauss_comb, init_values=(0.25, 10, 1.7, 0, 1.9, 30, 17.6, 0, 1.9, 30, 17.6, 0) ) radians : bool, optional Defaults to  False "
},
{
"ref":"hrosailing.pipeline.CurveExtension.process",
"url":2,
"doc":"Parameters      w_pts : WeightedPoints Returns    - pd : PolarDiagramCurve",
"func":1
},
{
"ref":"hrosailing.pipeline.PointcloudExtension",
"url":2,
"doc":"Parameters      sampler : Sampler, optional Defaults to UniformRandomSampler(2000) neighbourhood : Neighbourhood, optional Defaults to Ball() interpolator : Interpolator, optional Defaults to ArithmeticMeanInterpolator(50)"
},
{
"ref":"hrosailing.pipeline.PointcloudExtension.process",
"url":2,
"doc":"Parameters      w_pts : WeightedPoints Returns    - pd : PolarDiagramPointcloud",
"func":1
},
{
"ref":"hrosailing.pipeline.InterpolationWarning",
"url":2,
"doc":"Warning raised if neighbourhood is too small for successful interpolation"
},
{
"ref":"hrosailing.pipelinecomponents",
"url":3,
"doc":"Components for the PolarPipeline and PipelineExtension classes among other things."
},
{
"ref":"hrosailing.pipelinecomponents.modelfunctions",
"url":4,
"doc":"Model functions that can be used with the Regressor class to model certain ship behaviours"
},
{
"ref":"hrosailing.pipelinecomponents.modelfunctions.ws_times_wa",
"url":4,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.modelfunctions.ws_concave_dt_wa",
"url":4,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.modelfunctions.ws_wa_s_dt",
"url":4,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.modelfunctions.ws_s_dt_wa_gauss",
"url":4,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.modelfunctions.ws_s_s_dt_wa_gauss_comb",
"url":4,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.modelfunctions.ws_s_wa_gauss",
"url":4,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.regressor",
"url":5,
"doc":"Contains the baseclass for Regressors used in the CurveExtension class, that can also be used to create custom Regressors. Also contains two predefined and usable regressors, the ODRegressor and the LeastSquareRegressor."
},
{
"ref":"hrosailing.pipelinecomponents.regressor.RegressorException",
"url":5,
"doc":"Custom exception for errors that may appear whilst working with the Regressor class and subclasses"
},
{
"ref":"hrosailing.pipelinecomponents.regressor.Regressor",
"url":5,
"doc":"Base class for all regressor classes Abstract Methods         model_func optimal_params set_weights(self, X_weights, y_weights) fit(self, data)"
},
{
"ref":"hrosailing.pipelinecomponents.regressor.Regressor.model_func",
"url":5,
"doc":"This property should return a version of the in the regression used model function"
},
{
"ref":"hrosailing.pipelinecomponents.regressor.Regressor.optimal_params",
"url":5,
"doc":"This property should return a version of the through regression determined optimal parameters of the model function"
},
{
"ref":"hrosailing.pipelinecomponents.regressor.Regressor.fit",
"url":5,
"doc":"This method should, given data, be used to determine optimal parameters for the model function",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.regressor.ODRegressor",
"url":5,
"doc":"An orthogonal distance regressor based on scipy.odr.odrpack Parameters      model_func : function The function which describes the model and is to be fitted. The function signature should be f(ws, wa,  params) -> bsp, where ws and wa are numpy.ndarrays resp. and params is a sequence of parameters that will be fitted init_values : array_like, optional Inital guesses for the optimal parameters of model_func that are passed to the scipy.odr.ODR class Defaults to None max_it : int, optional Maximum number of iterations done by scipy.odr.ODR. Defaults to 1000"
},
{
"ref":"hrosailing.pipelinecomponents.regressor.ODRegressor.model_func",
"url":5,
"doc":"Returns a read-only version of self._func"
},
{
"ref":"hrosailing.pipelinecomponents.regressor.ODRegressor.optimal_params",
"url":5,
"doc":"Returns a read-only version of self._popt"
},
{
"ref":"hrosailing.pipelinecomponents.regressor.ODRegressor.fit",
"url":5,
"doc":"Fits the model function to the given data, ie calculates the optimal parameters to minimize an objective function based on the data, see also [ODRPACK](https: docs.scipy.org/doc/external/odrpack_guide.pdf) Parameters      data : array_like of shape (n, 3) Data to which the model function will be fitted, given as a sequence of points consisting of wind speed, wind angle and boat speed",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.regressor.LeastSquareRegressor",
"url":5,
"doc":"A least square regressor based on scipy.optimize.curve_fit Parameters      model_func : function or callable The function which describes the model and is to be fitted. The function signature should be f(ws, wa,  params) -> bsp, where ws and wa are numpy.ndarrays resp. and params is a sequence of parameters that will be fitted init_vals : array_like ,optional Inital guesses for the optimal parameters of model_func that are passed to scipy.optimize.curve_fit Defaults to None"
},
{
"ref":"hrosailing.pipelinecomponents.regressor.LeastSquareRegressor.model_func",
"url":5,
"doc":"Returns a read-only version of self._func"
},
{
"ref":"hrosailing.pipelinecomponents.regressor.LeastSquareRegressor.optimal_params",
"url":5,
"doc":"Returns a read-only version of self._popt"
},
{
"ref":"hrosailing.pipelinecomponents.regressor.LeastSquareRegressor.fit",
"url":5,
"doc":"Fits the model function to the given data, ie calculates the optimal parameters to minimize the sum of the squares of the residuals, see also [curve_fit]( https: docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html ) Parameters      data : array_like of shape (n, 3) Data to which the model function will be fitted, given as a sequence of points consisting of wind speed, wind angle and boat speed",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.interpolator",
"url":6,
"doc":"Contains the baseclass for Interpolators used in the TableExtension and PointcloudExtension class, that can also be used to create custom Interpolators. Also contains various predefined and usable interpolators"
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.scaled",
"url":6,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.euclidean_norm",
"url":6,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.InterpolatorInitializationException",
"url":6,
"doc":"Exception raised if an error occurs during initialization of an Interpolator"
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.Interpolator",
"url":6,
"doc":"Base class for all Interpolator classes Abstract Methods         interpolate(self, w_pts)"
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.Interpolator.interpolate",
"url":6,
"doc":"This method should be used, given a point grid_pt and an instances of WeightedPoints, to determine the z-value at grid_pt, based on the z-values of the points in the WeightedPoints instance",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.IDWInterpolator",
"url":6,
"doc":"Basic inverse distance interpolator, based on the work of Shepard, \"A two-dimensional interpolation function for irregulary-spaced data\" For a given point grid_pt, that is to be interpolated, we calculate the distances d_pt =  grid-pt - pt[:2] for all considered measured points. Then we set the weights of a point pt to be w_pt = 1 / d_pt^p, for some nonnegative integer p The interpolated value on grid_pt then equals (\u03a3 w_pt pt[2]) / \u03a3 w_pt or if grid_pt is already a measured point pt , it will equal pt [2] Parameters      p : nonnegative int, optional Defaults to 2 norm : function or callable, optional Norm with which to calculate the distances, ie  . If nothing is passed, it will default to a scaled version of  . _2 Raises an InterpolatorInitializationException if p is negative"
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.IDWInterpolator.interpolate",
"url":6,
"doc":"Interpolates a given grid_pt according to the above described method Parameters      w_pts : WeightedPoints Considered measured points grid_pt : numpy.ndarray of shape (2,) Point that is to be interpolated Returns    - out : int / float Interpolated values at grid_pt",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.gauss_potential",
"url":6,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.ArithmeticMeanInterpolator",
"url":6,
"doc":"An Interpolator that gets the interpolated value according to the following procedure First the distance of the independent variables of all considered points and of the to interpolate point is calculated, ie  p[:2] - grid_pt  Then using a distribution, new weights are calculated based on the old weights, the previously calculated distances and other parameters depending on the distribution The value of the dependent variable of the interpolated point then equals s  (\u03a3 w_p  p) / \u03a3 w_p where s is an additional scaling factor In fact it is a more general approach to the inverse distance interpolator Parameters      s : positive int or float, optional Scaling factor for the arithmetic mean, Defaults to 1 norm : function or callable, optional Norm with which to calculate the distances, ie  . If nothing is passed, it will default to a scaled version of  . _2 distribution : function or callable, optional Function with which to calculate the updated weights. Should have the signature f(distances, old_weights,  parameters) -> new_weights If nothing is passed, it will default to gauss_potential, which calculated weights based on the formula \u03b2  exp(-\u03b1  old_weights  distances) params: Parameters to be passed to distribution Raises an InterpolatorInitializationException if s is nonpositive"
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.ArithmeticMeanInterpolator.interpolate",
"url":6,
"doc":"Interpolates a given grid_pt according to the above described method Parameters      w_pts : WeightedPoints Considered measured points grid_pt : numpy.ndarray of shape (2,) Point that is to be interpolated Returns    - out : int / float Interpolated values at grid_pt",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.ImprovedIDWInterpolator",
"url":6,
"doc":"An improved inverse distance interpolator, based on the work of Shepard, \"A two-dimensional interpolation function for irregulary-spaced data\" Instead of setting the weights as the normal inverse distance to some power, we set the weights in the following way: Let r be the radius of the ScalingBall with the center being some point grid_pt which is to be interpolated. For all considered measured points let d_pt be the same as in IDWInterpolator. If d_pt <= r/3 we set w_pt = 1 / d_pt. Otherwise we set w_pt = 27 / (4  r)  (d_pt / r - 1)^2 The resulting value on grid_pt will then be calculated the same way as in IDWInterpolator Parameters      norm : function or callable, optional Norm with which to calculate the distances, ie  . If nothing is passed, it will default to a scaled version of  . _2"
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.ImprovedIDWInterpolator.interpolate",
"url":6,
"doc":"Interpolates a given grid_pt according to the above described method Parameters      w_pts : WeightedPoints Considered measured points grid_pt : numpy.ndarray of shape (2,) Point that is to be interpolated Returns    - out : int / float Interpolated values at grid_pt",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.ShepardInterpolator",
"url":6,
"doc":"A full featured inverse distance interpolator, based on the work of Shepard, \"A two-dimensional interpolation function for irregulary-spaced data\" Parameters      tol : positive float , optional Defautls to numpy.finfo(float).eps slope: positive float, optional Defaults to 0.1 norm : function or callable, optional Norm with which to calculate the distances, ie  . If nothing is passed, it will default to a scaled version of  . _2 Raises an InterpolatorInitializationException - if tol is nonpositive - if slope is nonpositive"
},
{
"ref":"hrosailing.pipelinecomponents.interpolator.ShepardInterpolator.interpolate",
"url":6,
"doc":"Interpolates a given grid_pt according to the above described method Parameters      w_pts : WeightedPoints Considered measured points grid_pt : numpy.ndarray of shape (2,) Point that is to be interpolated Returns    - out : int / float Interpolated values at grid_pt",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.filter",
"url":7,
"doc":"Contains the baseclass for Filters used in the PolarPipeline class, that can also be used to create custom Filters. Also contains two predefinied and usable filters, the QuantileFilter and the BoundFilter."
},
{
"ref":"hrosailing.pipelinecomponents.filter.FilterInitializationException",
"url":7,
"doc":"Exception raised if an error occurs during initialization of a Filter"
},
{
"ref":"hrosailing.pipelinecomponents.filter.Filter",
"url":7,
"doc":"Base class for all filter classes Abstract Methods         filter(self, weights)"
},
{
"ref":"hrosailing.pipelinecomponents.filter.Filter.filter",
"url":7,
"doc":"This method should be used, given an array of weights, to filter out points based on their weights, and produce a boolean array of the same size as wts",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.filter.QuantileFilter",
"url":7,
"doc":"A filter that filteres all points based on if their resp. weight lies above a certain quantile Parameters      percent: int or float, optional The quantile to be calculated Defaults to 25 Raises a FilterInitializationException, if percent is not in the interval [0, 100]"
},
{
"ref":"hrosailing.pipelinecomponents.filter.QuantileFilter.filter",
"url":7,
"doc":"Filters a set of points given by their resp. weights according to the above described method Parameters      wts : numpy.ndarray of shape (n, ) Weights of the points that are to be filtered, given as a sequence of scalars Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing with points are filtered depending on their resp. weight",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.filter.BoundFilter",
"url":7,
"doc":"A filter that filters all points based on if their weight is outside an interval given by a lower and upper bound Parameters      upper_bound : int or float, optional The upper bound for the filter Defaults to 1 lower_bound : int or float, optional The lower bound for the filter Defaults to 0.5 Raises a FilterInitializationException if lower_bound is greater than upper_bound"
},
{
"ref":"hrosailing.pipelinecomponents.filter.BoundFilter.filter",
"url":7,
"doc":"Filters a set of points given by their resp. weights according to the above described method Parameters      wts : numpy.ndarray of shape (n, ) Weights of the points that are to be filtered, given as a sequence of scalars Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing with points are filtered depending on their resp. weight",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.influencemodel",
"url":8,
"doc":""
},
{
"ref":"hrosailing.pipelinecomponents.influencemodel.InfluenceModel",
"url":8,
"doc":""
},
{
"ref":"hrosailing.pipelinecomponents.influencemodel.InfluenceModel.remove_influence",
"url":8,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.influencemodel.InfluenceModel.add_influence",
"url":8,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.influencemodel.LinearCurrentModel",
"url":8,
"doc":""
},
{
"ref":"hrosailing.pipelinecomponents.influencemodel.LinearCurrentModel.remove_influence",
"url":8,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.influencemodel.LinearCurrentModel.add_influence",
"url":8,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.weigher",
"url":9,
"doc":"Contains the baseclass for Weighers used in the PolarPipeline class, that can also be used to create custom Weighers. Also contains two predefined and useable weighers, the CylindricMeanWeigher and the CylindricMemberWeigher, aswell as the WeightedPoints class, used to represent data points together with their respective weights"
},
{
"ref":"hrosailing.pipelinecomponents.weigher.scaled",
"url":9,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.weigher.euclidean_norm",
"url":9,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.weigher.WeigherInitializationException",
"url":9,
"doc":"Exception raised if an error occurs during initialization of a Weigher"
},
{
"ref":"hrosailing.pipelinecomponents.weigher.WeighingException",
"url":9,
"doc":"Exception raised if an error occurs during the calling of the .weigh() method"
},
{
"ref":"hrosailing.pipelinecomponents.weigher.Weigher",
"url":9,
"doc":"Base class for all weigher classes Abstract Methods         weight(self, pts)"
},
{
"ref":"hrosailing.pipelinecomponents.weigher.Weigher.weigh",
"url":9,
"doc":"This method should be used, given certain points, to determine their weights according to a weighing method",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.weigher.CylindricMeanWeigher",
"url":9,
"doc":"A weigher that weighs given points according to the following procedure: For a given point p and points pts we look at all the points pt in pts such that  pt[:d-1] - p[:d-1] <= r Then we take the mean m_p and standard deviation std_p of the dth component of all those points and set w_p = | m_p - p[d-1] | / std_p Parameters      radius : positive int or float, optional The radius of the considered cylinder, with infinite height, ie r Defaults to 0.05 norm : function or callable, optional Norm with which to evaluate the distances, ie  . If nothing is passed, it will default to  . _2 Raises a WeigherInitializationException if radius is nonpositive"
},
{
"ref":"hrosailing.pipelinecomponents.weigher.CylindricMeanWeigher.weigh",
"url":9,
"doc":"Weigh given points according to the method described above Parameters      pts : numpy.ndarray of shape (n, 3) Points to be weight Returns    - wts : numpy.ndarray of shape (n, ) Normalized weights of the input points",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.weigher.CylindricMemberWeigher",
"url":9,
"doc":"A weigher that weighs given points according to the following procedure: For a given point p and points pts we look at all the points pt in pts such that |pt[0] - p[0]| <= h and  pt[1:] - p[1:] <= r Call the set of all such points P, then w_p =  P - 1 Parameters      radius : positive int or float, optional The radius of the considered cylinder, ie r Defaults to 0.05 length : nonnegative int of float, optional The height of the considered cylinder, ie h If length is 0, the cylinder is a d-1 dimensional ball Defaults to 0.05 norm : function or callable, optional Norm with which to evaluate the distances, ie  . If nothing is passed, it will default to  . _2 Raises a WeigherInitializationException - if radius is nonpositive - if length is negative"
},
{
"ref":"hrosailing.pipelinecomponents.weigher.CylindricMemberWeigher.weigh",
"url":9,
"doc":"Weigh given points according to the method described above Parameters      pts : numpy.ndarray of shape (n, 3) Points to be weight Returns    - wts : numpy.ndarray of shape (n, ) Normalized weights of the input points",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.weigher.WeightedPoints",
"url":9,
"doc":"A class to weigh data points and represent them together with their respective weights Parameters      pts : array_like of shape (n, 3) Points that will be weight or paired with given weights wts : int, float or array_like of shape (n, ), optional If the weights of the points are known beforehand, they can be given as an argument. If weights are passed, they will be assigned to the points and no further weighing will take place If a scalar is passed, the points will all be assigned the same weight Defaults to None weigher : Weigher, optional Instance of a Weigher class, which will weigh the points Will only be used if weights is None If nothing is passed, it will default to CylindricMeanWeigher() tw : bool, optional Specifies if the given wind data should be viewed as true wind If False, wind data will be converted to true wind Defaults to True"
},
{
"ref":"hrosailing.pipelinecomponents.weigher.WeightedPoints.points",
"url":9,
"doc":"Returns a read-only version of self._pts"
},
{
"ref":"hrosailing.pipelinecomponents.weigher.WeightedPoints.weights",
"url":9,
"doc":"Returns a read-only version of self._wts"
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood",
"url":10,
"doc":"Contains the baseclass for Neighbourhoods used in the TableExtension and PointcloudExtension class, that can also be used to create custom Neighbourhoods. Also contains various predefined and usable neighbourhoods"
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.scaled",
"url":10,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.euclidean_norm",
"url":10,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.NeighbourhoodInitializationException",
"url":10,
"doc":"Exception raised if an error occurs during initialization of a Neighbourhood"
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.Neighbourhood",
"url":10,
"doc":"Base class for all neighbourhood classes Abstract Methods         is_contained_in(self, pts)"
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.Neighbourhood.is_contained_in",
"url":10,
"doc":"This method should be used, given certain points, to determine which of these points lie in the neighbourhood and which do not, by producing a boolean array of the same size as pts",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.Ball",
"url":10,
"doc":"A class to describe a closed 2-dimensional ball centered around the origin, ie { x in R^2 :  x <= r } Parameters      norm : function or callable, optional The norm for which the ball is described, ie  . If nothing is passed, it will default to a scaled version of  . _2 radius : positive int or float, optional The radius of the ball, ie r Defaults to 0.05 Raises a NeighbourhoodInitializationException if radius is nonpositive"
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.Ball.is_contained_in",
"url":10,
"doc":"Checks given points for membership. Parameters      pts : array_like of shape (n, 2) Points that will be checked for membership Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing which of the input points is a member of the neighbourhood",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.ScalingBall",
"url":10,
"doc":"A class to represent a closed 2-dimensional ball centered around the origin, ie { x in R^2 :  x <= r }, where the radius r will be dynamically determined, such that there are always a certain amount of given points contained in the ball Parameters      min_pts : positive int The minimal amount of certain given points that should be contained in the scaling ball max_pts : positive int The \"maximal\" amount of certain given points that should be contained in the scaling ball. Mostly used for initial guess of a \"good\" radius. Also to guarantee that on average, the scaling ball will contain (min_pts + max_pts) / 2 points of certain given points It is also unlikely that the scaling ball will contain more than max_pts points norm : function or callable, optional The norm for which the scaling ball is described, ie  . If nothing is passed, it will default to a scaled version of  . _2 Raises a NeighbourhoodInitializationException - if min_pts or max_pts are nonpositive - if max_pts is less than or equal to min_pts"
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.ScalingBall.is_contained_in",
"url":10,
"doc":"Checks given points for membership, and scales ball so that at least min_pts points are contained in it Parameters      pts : array_like of shape (n, 2) Points that will be checked for membership Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing which of the input points is a member of the neighbourhood",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.Ellipsoid",
"url":10,
"doc":"A class to represent a closed d-dimensional ellipsoid centered around the origin, ie T(B), where T is an invertible linear transformation, and B is a closed d-dimensional ball, centered around the origin. It will be represented using the equivalent formulation: { x in R^2 :  T^-1 x <= r } Parameters      lin_trans: array_like of shape (2,2), optional The linear transformation which transforms the ball into the given ellipsoid, ie T If nothing is passed, it will default to I_2, the 2x2 unit matrix, ie the ellipsoid will be a ball norm : function or callable, optional The norm for which the ellipsoid is described, ie  . If nothing is passed, it will default to a scaled version of  . _2 radius : positive int or float, optional The radius of the ellipsoid, ie r Defaults to 0.05 Raises a NeighbourhoodInitializationException - if radius is nonpositive - if lin_trans is not a (2,2)-array or is not invertible"
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.Ellipsoid.is_contained_in",
"url":10,
"doc":"Checks given points for membership. Parameters      pts : array_like of shape (n, 2) Points that will be checked for membership Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing which of the input points is a member of the neighbourhood",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.Cuboid",
"url":10,
"doc":"A class to represent a d-dimensional closed cuboid, ie { x in R^2 : |x_i| <= b_i, i=1,2 } Parameters      norm : function or callable, optional The 1-d norm used to measure the length of the x_i, ie |.| If nothing is passed, it will default to the absolute value |.| dimensions: subscriptable of length 2, optional The 'length' of the 'sides' of the cuboid, ie the b_i If nothing is passed, it will default to (0.05, 0.05)"
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.Cuboid.is_contained_in",
"url":10,
"doc":"Checks given points for membership. Parameters      pts : array_like of shape (n, 2) Points that will be checked for membership Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing which of the input points is a member of the neighbourhood",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.Polytope",
"url":10,
"doc":"A class to represent a general 2-dimensional polytope, ie the convex hull P = conv(x_1,  ., x_n) of some n points x_1 , ., x_n or equivalent as the (bounded) intersection of m half spaces P = { x in R^2 : Ax <= b } Parameters      mat: array_like of shape (m, 2), optional matrix to represent the normal vectors a_i of the half spaces, ie A = (a_1,  . , a_m)^t If nothing is passed, it will default to (I_2, -I_2)^t, where I_d is the d-dimensional unit matrix b: array_like of shape (m, ), optional vector to represent the  . b_i of the half spaces, ie b = (b_1,  . , b_m)^t If nothing is passed, it will default to (0.05, .,0.05) Raises a NeighbourhoodException if mat and b are not of matching shape Warning    - Does not check wether the polytope given by mat and b is a polytope, ie if P is actually bounded"
},
{
"ref":"hrosailing.pipelinecomponents.neighbourhood.Polytope.is_contained_in",
"url":10,
"doc":"Checks given points for membership. Parameters      pts : array_like of shape (n, 2) Points that will be checked for membership Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing which of the input points is a member of the neighbourhood",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.datahandler",
"url":11,
"doc":""
},
{
"ref":"hrosailing.pipelinecomponents.datahandler.HandlerInitializationException",
"url":11,
"doc":"Exception raised if an error occurs during initialization of a DataHandler"
},
{
"ref":"hrosailing.pipelinecomponents.datahandler.HandleException",
"url":11,
"doc":"Exception raised if an error occurs during calling of the .handle() method"
},
{
"ref":"hrosailing.pipelinecomponents.datahandler.DataHandler",
"url":11,
"doc":"Base class for all datahandler classes Abstract Methods         handle(self, data)"
},
{
"ref":"hrosailing.pipelinecomponents.datahandler.DataHandler.handle",
"url":11,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.datahandler.ArrayHandler",
"url":11,
"doc":"A data handler to handle data, given as an array_like sequence. Doesn't really do anything since error handling and array conversion is handeled by the pipeline itself Only needed for general layout of the pipeline"
},
{
"ref":"hrosailing.pipelinecomponents.datahandler.ArrayHandler.handle",
"url":11,
"doc":"",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.datahandler.CsvFileHandler",
"url":11,
"doc":"A data handler to extract data from a .csv file with the first three columns representing wind speed, wind angle, and boat speed respectively"
},
{
"ref":"hrosailing.pipelinecomponents.datahandler.CsvFileHandler.handle",
"url":11,
"doc":"Reads a .csv file and extracts the contained data points The delimiter used in the .csv file Parameters      data : path-like Path to a .csv file Returns    -",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.datahandler.NMEAFileHandler",
"url":11,
"doc":"A data handler to extract data from a text file containing certain nmea sentences Parameters     - sentences : Iterable of Strings, attributes : Iterable of Strings, mode : string, optional In the case where there is more recorded wind data than speed data, specifies how to handle the surplus - \"interpolate\": handles the surplus by taking convex combinations of two recorded speed datas together with the recorded wind data \"between\" those two points to create multiple data points - \"mean\": handles the surplus by taking the mean of the wind data \"belonging\" to a given speed data to create a singe data point Defaults to \"interpolate\" Raises a HandlerInitializationException if mode is not one of the above choices"
},
{
"ref":"hrosailing.pipelinecomponents.datahandler.NMEAFileHandler.handle",
"url":11,
"doc":"Reads a text file containing nmea-sentences and extracts data points Parameters      data : path-like Path to a text file, containing nmea-0183 sentences Returns    - data_dict : dict",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.sampler",
"url":12,
"doc":"Contains the baseclass for Samplers used in the PointcloudExtension class, that can also be used to create custom Samplers. Also contains various predefined and usable samplers."
},
{
"ref":"hrosailing.pipelinecomponents.sampler.SamplerInitializationException",
"url":12,
"doc":"Exception raised if an error occurs during initialization of a Sampler"
},
{
"ref":"hrosailing.pipelinecomponents.sampler.Sampler",
"url":12,
"doc":"Base class for all sampler classes Abstract Methods         sample(self, pts)"
},
{
"ref":"hrosailing.pipelinecomponents.sampler.Sampler.sample",
"url":12,
"doc":"This method should be used, given certain points, to determine a constant number of sample points that lie in the convex hull of pts and are more or less representative of the trend of the given points",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.sampler.UniformRandomSampler",
"url":12,
"doc":"A sampler that produces a number of uniformly distributed samples, which all lie in the convex hull of certain given points Parameters      n_samples : positive int Amount of samples that will be produced by the sampler Raises SamplerInitializationException if n_samples is nonpositive"
},
{
"ref":"hrosailing.pipelinecomponents.sampler.UniformRandomSampler.sample",
"url":12,
"doc":"Produces samples according to the above described procedure Parameters      pts : array_like of shape (n, 2) Points in whose convex hull the produced samples will lie Returns    - samples : numpy.ndarray of shape (n_samples, 2) samples produced by the above described method",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.sampler.FibonacciSampler",
"url":12,
"doc":"A sampler that produces sample points on a moved and scaled version of the spiral (sqrt(x) cos(x), sqrt(x) sin(x , such that the angles are distributed equidistantly by the inverse golden ratio. The sample points all lie in the smallest enclosing circle of given data points. Inspired by \u00c1lvaro Gonzz\u00e1lez - \"Measurement of areas on a sphere using Fibonacci and latitude\u2013longitude lattices\" Parameters      n_samples : positive int Amount of samples that will be produced by the sampler Raises SamplerInitializationException if n_samples is nonpositive"
},
{
"ref":"hrosailing.pipelinecomponents.sampler.FibonacciSampler.sample",
"url":12,
"doc":"Produces samples according to the above described procedure Parameters      pts : array_like of shape (n, 2) Points in whose convex hull the produced samples will lie Returns    - samples : numpy.ndarray of shape (n_samples, 2) samples produced by the above described method",
"func":1
},
{
"ref":"hrosailing.pipelinecomponents.sampler.ArchimedianSampler",
"url":12,
"doc":"A sampler that produces a number of approximately equidistant sample points on a moved and scaled version of the archimedean spiral (x cos(x), x sin(x . The sample points all lie in the smallest enclosing circle of given data points. Inspired by https: agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007GC001581 Parameters      n_samples : positive int Amount of samples that will be produced by the sampler Raises SamplerInitializationException if n_samples is nonpositive"
},
{
"ref":"hrosailing.pipelinecomponents.sampler.ArchimedianSampler.sample",
"url":12,
"doc":"Produces samples according to the above described procedure Parameters      pts : array_like of shape (n, 2) Points in whose convex hull the produced samples will lie Returns    - samples : numpy.ndarray of shape (n_samples, 2) samples produced by the above described method",
"func":1
},
{
"ref":"hrosailing.wind",
"url":13,
"doc":"Functions to convert wind from apparent to true and vice versa"
},
{
"ref":"hrosailing.wind.WindConversionException",
"url":13,
"doc":"Exception raised if an error occurs during wind conversion"
},
{
"ref":"hrosailing.wind.apparent_wind_to_true",
"url":13,
"doc":"Converts apparent wind to true wind Parameters      wind : array_like of shape (n, 3) Wind data given as a sequence of points consisting of wind speed, wind angle and boat speed, where the wind speed and wind angle are measured as apparent wind Returns    - out : numpy.ndarray of shape (n, 3) Array containing the same data as wind_arr, but the wind speed and wind angle now measured as true wind",
"func":1
},
{
"ref":"hrosailing.wind.true_wind_to_apparent",
"url":13,
"doc":"Converts true wind to apparent wind Parameters      wind : array_like of shape (n, 3) Wind data given as a sequence of points consisting of wind speed, wind angle and boat speed, where the wind speed and wind angle are measured as true wind Returns    - out : numpy.ndarray of shape (n, 3) Array containing the same data as wind_arr, but the wind speed and wind angle now measured as apparent wind",
"func":1
},
{
"ref":"hrosailing.cruising",
"url":14,
"doc":"Functions for navigation and weather routing using PPDs"
},
{
"ref":"hrosailing.cruising.Direction",
"url":14,
"doc":"Dataclass to represent sections of a sailing maneuver"
},
{
"ref":"hrosailing.cruising.Direction.angle",
"url":14,
"doc":""
},
{
"ref":"hrosailing.cruising.Direction.proportion",
"url":14,
"doc":""
},
{
"ref":"hrosailing.cruising.Direction.sail",
"url":14,
"doc":""
},
{
"ref":"hrosailing.cruising.convex_direction",
"url":14,
"doc":"Parameters      pd : PolarDiagram The polar diagram of the vessel ws : int / float The current wind speed given in knots direction : int / float Angle to the wind direction im : InfluenceModel, optional The influence model used to consider additional influences on the boat speed Defaults to  None influence_data: dict, optional Data containing information that might influence the boat speed of the vessel (eg. current, wave height), to be passed to the used influence model Only used, if  im is not  None Defaults to  None Returns    - edge : list of Directions of length 2",
"func":1
},
{
"ref":"hrosailing.cruising.cruise",
"url":14,
"doc":"Given a starting point A and and end point B,the function calculates the fastest time and sailing direction it takes for a sailing vessel to reach B from A, under constant wind. If needed the function will calculate two directions as well as the time needed to sail in each direction to get to B. Parameters      pd : PolarDiagram The polar diagram of the vessel ws : int / float The current wind speed given in knots wdir : The direction of the wind given as either - the wind angle relative to north - the true wind angle and the boat direction relative to north - the apparent wind angle and the boat direction relative to north - a (ugrd, vgrd) tuple from grib data start : tuple of length 2 Coordinates of the starting point of the cruising maneuver, given in longitude and latitude end : tuple of length 2 Coordinates of the end point of the cruising maneuver, given in longitude and latitude im : InfluenceModel, optional The influence model used to consider additional influences on the boat speed Defaults to  None influence_data: dict, optional Data containing information that might influence the boat speed of the vessel (eg. current, wave height), to be passed to the used influence model Only used, if  im is not  None Defaults to  None Returns    - out : list of tuples Directions as well as the time needed to sail along those, to get from start to end",
"func":1
},
{
"ref":"hrosailing.cruising.WeatherException",
"url":14,
"doc":""
},
{
"ref":"hrosailing.cruising.WeatherModel",
"url":14,
"doc":""
},
{
"ref":"hrosailing.cruising.WeatherModel.get_weather",
"url":14,
"doc":"",
"func":1
},
{
"ref":"hrosailing.cruising.cost_cruise",
"url":14,
"doc":"Computes the total cost for traveling from a start position to an end position To be precise, it calculates for a given cost density function cost and absolute function abs_cost int_0^l cost(s, t(s ds + abs_cost(t(l), l), where s is the distance travelled, l is the total distance from start to end and t(s) is the time travelled t(s) is the solution of the initial value problem t(0) = 0, dt/ds = 1/bsp(s,t). The costs also depend on the weather forecast data, organized by a WeatherModel Distances are computed using the mercator projection Parameter      pd: PolarDiagram Polar diagram of the vessel start: tuple of two floats Coordinates of the starting point end: tuple of two floats Coordinates of the end point start_time: datetime.datetime The time at which the traveling starts wm: WeatherModel, optional The WeatherModel used cost_fun_dens: callable, optional Function giving a cost density for given time as datetime.datetime, lattitude as float, longitude as float and WeatherModel cost_fun_dens(t,lat,long,wm) corresponds to costs(s,t) above. Defaults to None. cost_fun_abs: callable, optional Corresponds to abs_costs above. Defaults to lambda total_t, total_s: total_t integration_method: callable, optional Function that takes two (n,) arrays y, x and computes an approximative integral from that. Is only used if cost_fun_dens is not None Defaults to scipy.integrate.trapezoid im: InfluenceModel, optional The influence model used to consider additional influences on the boat speed. Defaults to  ivp_kw: Keyword arguments which will be redirected to scipy.integrate.solve_ivp in order to solve the initial value problem described above Returns    - out : float The total cost calculated as described above",
"func":1
},
{
"ref":"hrosailing.cruising.isocrone",
"url":14,
"doc":"Estimates the maximum distance that can be reached from a given start point in a given amount of time without tacks and jibes. This is done by sampling the position space and using mercator projection. A weather forecast, organized by a WeatherModel and an InfluenceModel are included in the computation. Parameter      pd: PolarDiagram The polar diagram of the used vessel start: 2-tuple of floats The lattitude and longitude of the starting point start_time: datetime.datetime The time at which the traveling starts direction: float The angle between North and the direction in which we aim to travel. wm: WeatherModel, optional The weather model used. total_time: float The time in hours that the vessel is supposed to travel in the given direction. min_nodes: int, optional The minimum amount of sample points to sample the position space. Defaults to 100. im: InfluenceModel, optional The influence model used. Defaults to  . Returns    - end : 2-tuple of floats Lattitude and Longitude of the position that is reached when traveling total_time hours in the given direction s : float The length of the way traveled from start to end",
"func":1
}
]