URLS=[
"hrosailing/index.html",
"hrosailing/polardiagram/index.html",
"hrosailing/wind.html",
"hrosailing/cruising/index.html",
"hrosailing/processing/index.html",
"hrosailing/processing/pipeline.html",
"hrosailing/processing/pipelinecomponents/index.html",
"hrosailing/processing/pipelinecomponents/regressor.html",
"hrosailing/processing/pipelinecomponents/interpolator.html",
"hrosailing/processing/pipelinecomponents/filter.html",
"hrosailing/processing/pipelinecomponents/influencemodel.html",
"hrosailing/processing/pipelinecomponents/weigher.html",
"hrosailing/processing/pipelinecomponents/neighbourhood.html",
"hrosailing/processing/pipelinecomponents/datahandler.html",
"hrosailing/processing/pipelinecomponents/sampler.html"
];
INDEX=[
{
"ref":"hrosailing",
"url":0,
"doc":""
},
{
"ref":"hrosailing.polardiagram",
"url":1,
"doc":"Classes to represent polar diagrams in various different forms as well as small functions to save / load PolarDiagram-objects to files in different forms and functions to manipulate PolarDiagram-objects"
},
{
"ref":"hrosailing.polardiagram.plotting",
"url":1,
"doc":"dummy"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramException",
"url":1,
"doc":"Custom exception for errors that may appear whilst handling polar diagrams"
},
{
"ref":"hrosailing.polardiagram.FileReadingException",
"url":1,
"doc":"Custom exception for errors that may appear whilst reading a file"
},
{
"ref":"hrosailing.polardiagram.FileWritingException",
"url":1,
"doc":"Custom exception for errors that may appear whilst writing to a file"
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
"doc":"Reads a .csv file and returns the PolarDiagram instance contained in it Parameters      csv_path : path-like Path to a .csv file fmt : string The \"format\" of the .csv file. - hro: format created by the to_csv-method of the PolarDiagram class - orc: format found at [ORC](https: jieter.github.io/orc-data/site/) - opencpn: format created by [OpenCPN Polar Plugin](https: opencpn.org/OpenCPN/plugins/polar.html) - array tw : bool Specifies if wind data in file should be viewed as true wind Defaults to True Returns    - out : PolarDiagram PolarDiagram instance contained in the .csv file Raises a FileReadingException - if an unknown format was specified - if an error occurs whilst reading",
"func":1
},
{
"ref":"hrosailing.polardiagram.pickling",
"url":1,
"doc":"See pickling()-method of PolarDiagram Parameters      pkl_path : path-like Path to a .pkl file or where a new .pkl file will be created obj : PolarDiagram PolarDiagram instance which will be written to .csv file Raises a FileWritingException if an error occurs whilst writing",
"func":1
},
{
"ref":"hrosailing.polardiagram.depickling",
"url":1,
"doc":"Reads a .pkl file and returns the PolarDiagram instance contained in it. Parameters      pkl_path : path-like Path to a .pkl file Returns    - out : PolarDiagram PolarDiagram instance contained in the .pkl file Raises a FileReadingException if an error occurs whilst reading",
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
"doc":"Base class for all polardiagram classes Abstract Methods         to_csv(csv_path) symmetrize() get_slices(ws) plot_polar( ws, ax=None, colors=(\"green\", \"red\"), show_legend=False, legend_kw=None,  plot_kw ) plot_flat( ws, ax=None, colors=(\"green\", \"red\"), show_legend=False, legend_kw=None,  plot_kw ) plot_3d(ax=None,  plot_kw) plot_color_gradient( ax=None, colors=(\"green\", \"red\"), marker=None, ms=None, show_legend=False,  legend_kw, ) plot_convex_hull( ws, ax=None, colors=(\"green\", \"red\"), show_legend=False, legend_kw=None,  plot_kw, )"
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.pickling",
"url":1,
"doc":"Writes PolarDiagram instance to a .pkl file Parameters      pkl_path: path-like Path to a .pkl file or where a new .pkl file will be created Raises a FileWritingException if an error occurs whilst writing",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.to_csv",
"url":1,
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.symmetrize",
"url":1,
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.get_slices",
"url":1,
"doc":"",
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
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_flat",
"url":1,
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_3d",
"url":1,
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagram.plot_color_gradient",
"url":1,
"doc":"",
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
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable",
"url":1,
"doc":"A class to represent, visualize and work with a polar diagram in the form of a table. Parameters      ws_res : array_like or positive int/float, optional Wind speeds that will correspond to the columns of the table Can either be a sequence of length cdim or an int/float value If a number num is passed, numpy.arange(num, 40, num) will be assigned to ws_res If nothing is passed, it will default to numpy.arange(2, 42, 2) wa_res : array_like or positive int/float, optional Wind angles that will correspond to the rows of the table. Should be between 0\u00b0 and 360\u00b0 Can either be sequence of length rdim or an int/float value If a number num is passed, numpy.arange(num, 360, num) will be assigned to wa_res If nothing is passed, it will default to numpy.arange(0, 360, 5) bsps : array_like, optional Boatspeeds that will correspond to the entries of the table Should be broadcastable to the shape (rdim, cdim) If nothing is passed it will default to numpy.zeros rdim, cdim Raises a PolarDiagramException - if bsps can't be broadcasted to a fitting shape - if bsps is not of dimension 2 - if bsps is an empty array Examples     >>> pd = PolarDiagramTable(ws_res = [6, 8, 10, 12, 14],  . wa_res = [52, 60, 75, 90, 110, 120, 135]) >>> print(pd) TWA \\ TWS 6.0 8.0 10.0 12.0 14.0      -   -   -          52.0 0.00 0.00 0.00 0.00 0.00 60.0 0.00 0.00 0.00 0.00 0.00 75.0 0.00 0.00 0.00 0.00 0.00 90.0 0.00 0.00 0.00 0.00 0.00 110.0 0.00 0.00 0.00 0.00 0.00 120.0 0.00 0.00 0.00 0.00 0.00 135.0 0.00 0.00 0.00 0.00 0.00 >>> pd = PolarDiagramTable() >>> pd.wind_speeds [ 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40] >>> pd.wind_angles [ 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 250 255 260 265 270 275 280 285 290 295 300 305 310 315 320 325 330 335 340 345 350 355]"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.__str__",
"url":1,
"doc":"Return str(self).",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.__repr__",
"url":1,
"doc":"Return repr(self).",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.__getitem__",
"url":1,
"doc":"",
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
"doc":"Creates a .csv file with delimiter ',' and the following format: PolarDiagramTable Wind speed resolution: self.wind_speeds Wind angle resolution: self.wind_angles Boat speeds: self.boat_speeds Parameters      csv_path : path-like Path to a .csv file or where a new .csv file will be created fmt : string Raises a FileWritingException - inputs are not of the specified types - if an error occurs whilst writing - unknown format was specified",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.symmetrize",
"url":1,
"doc":"Constructs a symmetric version of the polar diagram, by mirroring it at the 0\u00b0 - 180\u00b0 axis and returning a new instance Warning    - Should only be used if all the wind angles of the initial polar diagram are on one side of the 0\u00b0 - 180\u00b0 axis, otherwise this can lead to duplicate data, which can overwrite or live alongside old data Examples     >>> pd = PolarDiagramTable(ws_res = [6, 8, 10, 12, 14],  . wa_res = [52, 60, 75, 90, 110, 120, 135]) >>> sym_pd = pd.symmetrize() >>> print(sym_pd.wind_speeds) [ 6 8 10 12 14] >>> print(sym_pd.wind_angles) [ 52 60 75 90 110 120 135 225 240 250 270 285 300 308]",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.change_entries",
"url":1,
"doc":"Changes specified entries in the table Parameters      new_bsps: array_like of matching shape Sequence containing the new boat speeds to be inserted in the specified entries ws: Iterable or int or float, optional Element(s) of self.wind_speeds, specifying the columns, where new boat speeds will be inserted If nothing is passed it will default to self.wind_speeds wa: Iterable or int or float, optional Element(s) of self.wind_angles, specifiying the rows, where new boat speeds will be inserted If nothing is passed it will default to self.wind_angles Raises a PolarDiagramException",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.get_slices",
"url":1,
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_polar",
"url":1,
"doc":"Creates a polar plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram table, given as either - a tuple of length 2 specifying an interval of considered wind speeds - an iterable containing only elements of self.wind_speeds - a single element of self.wind_speeds The slices are then equal to the corresponding columns of the table together with self.wind_angles If nothing it passed, it will default to self.wind_speeds ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException - if at least one element of ws is not in self.wind_speeds - the given interval doesn't contain any slices of the polar diagram",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_flat",
"url":1,
"doc":"Creates a cartesian plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram table, given as either - a tuple of length 2 specifying an interval of considered wind speeds - an iterable containing only elements of self.wind_speeds - a single element of self.wind_speeds The slices are then equal to the corresponding columns of the table together with self.wind_angles If nothing it passed, it will default to self.wind_speeds ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException - if at least one element of ws is not in self.wind_speeds - the given interval doesn't contain any slices of the polar diagram",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_3d",
"url":1,
"doc":"Creates a 3d plot of the polar diagram Parameters      ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments, optional Keyword arguments to change certain appearences of the plot Only the  color / c keyword is used, the rest are only supported to be consistent with the plot_3d()-method of PolarDiagram and PolarDiagrampointcloud The value of the  color / c should be either a tuple or list of matplotlib supported color_like entries, if the 3d-plot should be plotted with a color gradient of those colors, or a single color_like value, if the 3d-plot should be plotted with a single color. If nothing is passed the  color / c keyword defaults to (\"green\", \"red\")",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramTable.plot_color_gradient",
"url":1,
"doc":"Creates a 'wind speed vs. wind angle' color gradient plot of the polar diagram with respect to the respective boat speeds Parameters      ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple of length 2, optional Colors which specify the color gradient with which the polar diagram will be plotted. Defaults to (\"green\", \"red\") marker : matplotlib.markers.Markerstyle or equivalent, optional Markerstyle for the created scatter plot If nothing is passed, it will default to \"o\" ms : float or array_like of fitting shape, optional Marker size in points 2 If nothing is passed, it will use the default of the matplotlib.pyplot.scatter function show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot Legend will be a matplotlib.colorbar.Colorbar object. Defaults to False legend_kw : Keyword arguments Keyword arguments to be passed to the matplotlib.colorbar.Colorbar class to change position and appearence of the legend. Will only be used if show_legend is True If nothing is passed, it will default to {}",
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
"doc":"Writes PolarDiagram instance to a .pkl file Parameters      pkl_path: path-like Path to a .pkl file or where a new .pkl file will be created Raises a FileWritingException if an error occurs whilst writing",
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
"ref":"hrosailing.polardiagram.interpolate",
"url":1,
"doc":"",
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
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.__str__",
"url":1,
"doc":"Return str(self).",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.__repr__",
"url":1,
"doc":"Return repr(self).",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.to_csv",
"url":1,
"doc":"",
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
"doc":"",
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
"doc":"Creates a 3d plot of the polar diagram Parameters      ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments, optional Keyword arguments to change certain appearences of the plot Only the  color / c keyword is used, the rest are only supported to be consistent with the plot_3d()-method of PolarDiagram and PolarDiagrampointcloud The value of the  color / c should be either a tuple or list of matplotlib supported color_like entries, if the 3d-plot should be plotted with a color gradient of those colors, or a single color_like value, if the 3d-plot should be plotted with a single color. If nothing is passed the  color / c keyword defaults to (\"green\", \"red\")",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramMultiSails.plot_color_gradient",
"url":1,
"doc":"",
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
"doc":"Writes PolarDiagram instance to a .pkl file Parameters      pkl_path: path-like Path to a .pkl file or where a new .pkl file will be created Raises a FileWritingException if an error occurs whilst writing",
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
"ref":"hrosailing.polardiagram.PolarDiagramCurve.__repr__",
"url":1,
"doc":"Return repr(self).",
"func":1
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
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.get_slices",
"url":1,
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_polar",
"url":1,
"doc":"Creates a polar plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram given as either - a tuple of length 2, specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of specific wind speeds - a single wind speed Slices will then equal self(w, wa) where w goes through the given values in  ws and wa goes through a fixed number of angles between 0\u00b0 and 360\u00b0 If nothing is passed, it will default to (0, 20) stepsize : positive int or float, optional Specfies the amount of slices taken from the given wind speed interval Will only be used if  ws is a tuple of length 2 If nothing is passed, it will default to ws[1] - ws[0] ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot",
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
"doc":"Creates a 3d plot of a part of the polar diagram Parameters      ws : tuple of length 2, optional A region of the polar diagram given as an interval of wind speeds Slices will then equal self(w, wa) where w goes through the given values in  ws and wa goes through a fixed number of angles between 0\u00b0 and 360\u00b0 If nothing is passed, it will default to (0, 20) stepsize : positive int or float, optional Specfies the amount of slices taken from the given interval in  ws If nothing is passed, it will default to 100 ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments, optional Keyword arguments to change certain appearences of the plot Only the  color / c keyword is used, the rest are only supported to be consistent with the plot_3d()-method of PolarDiagram and PolarDiagrampointcloud The value of the  color / c should be either a tuple or list of matplotlib supported color_like entries, if the 3d-plot should be plotted with a color gradient of those colors, or a single color_like value, if the 3d-plot should be plotted with a single color. If nothing is passed the  color / c keyword defaults to (\"green\", \"red\")",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_color_gradient",
"url":1,
"doc":"Creates a 'wind speed vs. wind angle' color gradient plot of a part of the polar diagram with respect to the respective boat speeds Parameters      ws : tuple of length 3, optional A region of the polar diagram given as an interval of wind speeds Slices will then equal self(w, wa) where w goes through the given values in  ws and wa goes through a fixed number of angles between 0\u00b0 and 360\u00b0 If nothing is passed, it will default to (0, 20) stepsize : positive int or float, optional Specfies the amount of slices taken from the given interval in  ws If nothing is passed, it will default to 100 ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple of length 2, optional Colors which specify the color gradient with which the polar diagram will be plotted. Defaults to (\"green\", \"red\") marker : matplotlib.markers.Markerstyle or equivalent, optional Markerstyle for the created scatter plot If nothing is passed, it will default to \"o\" ms : float or array_like of fitting shape, optional Marker size in points 2 If nothing is passed, it will use the default of the matplotlib.pyplot.scatter function show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot Legend will be a matplotlib.colorbar.Colorbar object. Defaults to False legend_kw : Keyword arguments Keyword arguments to be passed to the matplotlib.colorbar.Colorbar class to change position and appearence of the legend. Will only be used if show_legend is True If nothing is passed, it will default to {}",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.plot_convex_hull",
"url":1,
"doc":"Computes the (seperate) convex hull of one or more slices of the polar diagram and creates a polar plot of them Parameters      ws : tuple of length 2, iterable, int or float, optional Slices of the polar diagram given as either - a tuple of length 2, specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of specific wind speeds - a single wind speed Slices will then equal self(w, wa) where w goes through the given values in  ws and wa goes through a fixed number of angles between 0\u00b0 and 360\u00b0 If nothing is passed, it will default to (0, 20) stepsize : positive int or float, optional Specfies the amount of slices taken from the given wind speed interval Will only be used if  ws is a tuple of length 2 If nothing is passed, it will default to ws[1] - ws[0] ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramCurve.pickling",
"url":1,
"doc":"Writes PolarDiagram instance to a .pkl file Parameters      pkl_path: path-like Path to a .pkl file or where a new .pkl file will be created Raises a FileWritingException if an error occurs whilst writing",
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
"doc":"A class to represent, visualize and work with a polar diagram given by a point cloud Parameters      pts : array_like of shape (n, 3), optional Initial points of the point cloud, given as a sequence of points consisting of wind speed, wind angle and boat speed If nothing is passed, the point cloud will be initialized as an empty point cloud tw : bool, optional Specifies if the given wind data should be viewed as true wind If False, wind data will be converted to true wind Defaults to True Raises a PolarDiagramException - if - if"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.__str__",
"url":1,
"doc":"Return str(self).",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.__repr__",
"url":1,
"doc":"Return repr(self).",
"func":1
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
"doc":""
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.points",
"url":1,
"doc":"Returns a read only version of self._pts"
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.to_csv",
"url":1,
"doc":"Creates a .csv file with delimiter ',' and the following format: PolarDiagramPointcloud True wind speed ,True wind angle ,Boat speed self.points Parameters      csv_path : path-like Path to a .csv-file or where a new .csv file will be created Raises a FileWritingException if an error occurs whilst writing",
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
"doc":"Adds additional points to the point cloud Parameters      new_pts: array_like of shape (n, 3) New points to be added to the point cloud given as a sequence of points consisting of wind speed, wind angle and boat speed tw : bool, optional Specifies if the given wind data should be viewed as true wind If False, wind data will be converted to true wind Defaults to True Raises a PolarDiagramException",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.get_slices",
"url":1,
"doc":"",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_polar",
"url":1,
"doc":"Creates a polar plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable , int or float, optional Slices of the polar diagram given as either - a tuple of length 2 specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of tuples of length 2 and int/float values which will be interpreted as individual slices. If a w in  ws is an int or float, the given interval will be determined by the parameter  range_ . If it is a tuple, it will be interpreted as an inverval as is - a single wind speed. The given interval is then determined by the parameter  range_ A slice then consists of all rows in self.wind_speeds whose first entry lies in the interval given by w in  ws If nothing is passed, it will default to (min(self.wind_speeds), max(self.wind_speeds stepsize : positive int, optional Specfies the amount of slices taken from the given interval in  ws Will only be used if  ws is a tuple of length 2 If nothing is passed it will default to int(round(ws[1] - ws[0] range_ : positive int or float, optional Used to convert and int or float w in  ws to the interval (w - range_, w + range_ Will only be used if  ws is int or float or if any w in  ws is an int or float If nothing is passed, it will default to 1 ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException if ws is given as a single value or a list and there is a value w in ws, such that there are no rows in self.points whose first entry is equal to w",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_flat",
"url":1,
"doc":"Creates a cartesian plot of one or more slices of the polar diagram Parameters      ws : tuple of length 2, iterable , int or float, optional Slices of the polar diagram given as either - a tuple of length 2 specifying an interval of considered wind speeds. The amount of slices taken from that interval are determined by the parameter  stepsize - an iterable of tuples of length 2 and int/float values which will be interpreted as individual slices. If a w in  ws is an int or float, the given interval will be determined by the parameter  range_ . If it is a tuple, it will be interpreted as an inverval as is - a single wind speed. The given interval is then determined by the parameter  range_ A slice then consists of all rows in self.wind_speeds whose first entry lies in the interval given by w in  ws If nothing is passed, it will default to (min(self.wind_speeds), max(self.wind_speeds) stepsize : positive int, optional Specfies the amount of slices taken from the given interval in  ws Will only be used if  ws is a tuple of length 2 If nothing is passed it will default to int(round(ws[1] - ws[0] range_ : positive int or float, optional Used to convert and int or float w in  ws to the interval (w - range_, w + range_ Will only be used if  ws is int or float or if any w in  ws is an int or float If nothing is passed, it will default to 1 ax : matplotlib.projections.polar.PolarAxes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple, optional Specifies the colors to be used for the different slices. There are four options: - If as many or more colors as slices are passed, each slice will be plotted in the specified color - If exactly 2 colors are passed, the slices will be plotted with a color gradient consiting of the two colors - If more than 2 colors but less than slices are passed, the first n_color slices will be plotted in the specified colors, and the rest will be plotted in the default color \"blue\" - Alternatively one can specify certain slices to be plotted in a certain color by passing a tuple of (ws, color) pairs Defaults to (\"green\", \"red\") show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot The type of legend depends on the color options - If the slices are plotted with a color gradient, a matplotlib.colorbar.Colorbar object will be created and assigned to ax. - Otherwise a matplotlib.legend.Legend object will be created and assigned to ax. Defaults to False legend_kw : dict, optional Keyword arguments to be passed to either the matplotlib.colorbar.Colorbar or matplotlib.legend.Legend classes to change position and appearence of the legend Will only be used if show_legend is True If nothing is passed it will default to {} plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException if ws is given as a single value or a list and there is a value w in ws, such that there are no rows in self.points whose first entry is equal to w",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_3d",
"url":1,
"doc":"Creates a 3d plot of the polar diagram Parameters      ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes plot_kw : Keyword arguments Keyword arguments that will be passed to the matplotlib.axes.Axes.plot function, to change certain appearences of the plot Raises a PolarDiagramException if there are no points in the point cloud",
"func":1
},
{
"ref":"hrosailing.polardiagram.PolarDiagramPointcloud.plot_color_gradient",
"url":1,
"doc":"Creates a 'wind speed vs. wind angle' color gradient plot of the polar diagram with respect to the respective boat speeds Parameters      ax : matplotlib.axes.Axes, optional Axes instance where the plot will be created. If nothing is passed, the function will create a suitable axes colors : tuple of length 2, optional Colors which specify the color gradient with which the polar diagram will be plotted. Defaults to (\"green\", \"red\") marker : matplotlib.markers.Markerstyle or equivalent, optional Markerstyle for the created scatter plot If nothing is passed, it will default to \"o\" ms : float or array_like of fitting shape, optional Marker size in points 2 If nothing is passed, it will use the default of the matplotlib.pyplot.scatter function show_legend : bool, optional Specifies wether or not a legend will be shown next to the plot Legend will be a matplotlib.colorbar.Colorbar object. Defaults to False legend_kw : Keyword arguments Keyword arguments to be passed to the matplotlib.colorbar.Colorbar class to change position and appearence of the legend. Will only be used if show_legend is True If nothing is passed, it will default to {} Raises a PolarDiagramException if there are no points in the point cloud",
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
"doc":"Writes PolarDiagram instance to a .pkl file Parameters      pkl_path: path-like Path to a .pkl file or where a new .pkl file will be created Raises a FileWritingException if an error occurs whilst writing",
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
"ref":"hrosailing.wind",
"url":2,
"doc":"Functions to convert wind from apparent to true and vice versa"
},
{
"ref":"hrosailing.wind.WindException",
"url":2,
"doc":"Custom exception for errors that may appear during wind conversion or setting wind resolutions"
},
{
"ref":"hrosailing.wind.apparent_wind_to_true",
"url":2,
"doc":"Converts apparent wind to true wind Parameters      wind : array_like of shape (n, 3) Wind data given as a sequence of points consisting of wind speed, wind angle and boat speed, where the wind speed and wind angle are measured as apparent wind Returns    - out : numpy.ndarray of shape (n, 3) Array containing the same data as wind_arr, but the wind speed and wind angle now measured as true wind Raises a WindException - if wind is not of the specified type - if wind contains NaNs or infinite values",
"func":1
},
{
"ref":"hrosailing.wind.true_wind_to_apparent",
"url":2,
"doc":"Converts true wind to apparent wind Parameters      wind : array_like of shape (n, 3) Wind data given as a sequence of points consisting of wind speed, wind angle and boat speed, where the wind speed and wind angle are measured as true wind Returns    - out : numpy.ndarray of shape (n, 3) Array containing the same data as wind_arr, but the wind speed and wind angle now measured as apparent wind Raises a WindException - if wind is not of the specified type - if wind contains NaNs or infinite values",
"func":1
},
{
"ref":"hrosailing.wind.convert_wind",
"url":2,
"doc":"dummy"
},
{
"ref":"hrosailing.wind.set_resolution",
"url":2,
"doc":"dummy"
},
{
"ref":"hrosailing.cruising",
"url":3,
"doc":""
},
{
"ref":"hrosailing.cruising.convex_direction",
"url":3,
"doc":"",
"func":1
},
{
"ref":"hrosailing.cruising.cruise",
"url":3,
"doc":"",
"func":1
},
{
"ref":"hrosailing.cruising.WeatherModel",
"url":3,
"doc":""
},
{
"ref":"hrosailing.cruising.cost_cruise",
"url":3,
"doc":"",
"func":1
},
{
"ref":"hrosailing.cruising.isocrone",
"url":3,
"doc":"",
"func":1
},
{
"ref":"hrosailing.cruising.isocost",
"url":3,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing",
"url":4,
"doc":""
},
{
"ref":"hrosailing.processing.modelfunctions",
"url":4,
"doc":"dummy"
},
{
"ref":"hrosailing.processing.pipeline",
"url":5,
"doc":"A Pipeline class to automate getting a polar diagram from \"raw\" data"
},
{
"ref":"hrosailing.processing.pipeline.PipelineException",
"url":5,
"doc":""
},
{
"ref":"hrosailing.processing.pipeline.PipelineExtension",
"url":5,
"doc":""
},
{
"ref":"hrosailing.processing.pipeline.PipelineExtension.process",
"url":5,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipeline.PolarPipeline",
"url":5,
"doc":"A Pipeline class to create polar diagrams from raw data Parameters      extension: PipelineExtension handler : DataHandler weigher : Weigher, optional filter_ : Filter, optional Raises a PipelineException"
},
{
"ref":"hrosailing.processing.pipeline.PolarPipeline.__call__",
"url":5,
"doc":"Parameters      data : FooBar check_finite : bool, optional tw : bool, optional filtering : bool, optional Returns    - out : PolarDiagram",
"func":1
},
{
"ref":"hrosailing.processing.pipeline.TableExtension",
"url":5,
"doc":""
},
{
"ref":"hrosailing.processing.pipeline.TableExtension.process",
"url":5,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipeline.CurveExtension",
"url":5,
"doc":""
},
{
"ref":"hrosailing.processing.pipeline.CurveExtension.process",
"url":5,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipeline.PointcloudExtension",
"url":5,
"doc":""
},
{
"ref":"hrosailing.processing.pipeline.PointcloudExtension.process",
"url":5,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents",
"url":6,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor",
"url":7,
"doc":"Contains the baseclass for Regressors used in the CurveExtension class, that can also be used to create custom Regressors. Also contains two predefined and usable regressors, the ODRegressor and the LeastSquareRegressor."
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.RegressorException",
"url":7,
"doc":"Custom exception for errors that may appear whilst working with the Regressor class and subclasses"
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.Regressor",
"url":7,
"doc":"Base class for all regressor classes Abstract Methods         model_func optimal_params set_weights(self, X_weights, y_weights) fit(self, data)"
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.Regressor.model_func",
"url":7,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.Regressor.optimal_params",
"url":7,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.Regressor.fit",
"url":7,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.Regressor.set_weights",
"url":7,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.ODRegressor",
"url":7,
"doc":"An orthogonal distance regressor based on scipy.odr.odrpack Parameters      model_func : function The function which describes the model and is to be fitted. The function signature should be f(ws, wa,  params) -> bsp, where ws and wa are numpy.ndarrays resp. and params is a sequence of parameters that will be fitted init_values : array_like, optional Inital guesses for the optimal parameters of model_func that are passed to the scipy.odr.ODR class Defaults to None max_it : int, optional Maximum number of iterations done by scipy.odr.ODR. Defaults to 1000"
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.ODRegressor.model_func",
"url":7,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.ODRegressor.optimal_params",
"url":7,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.ODRegressor.set_weights",
"url":7,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.ODRegressor.fit",
"url":7,
"doc":"Fits the model function to the given data, ie calculates the optimal parameters to minimize an objective function based on the data, see also  ODRPACK   _ Parameters      data : array_like of shape (n, 3) Data to which the model function will be fitted, given as a sequence of points consisting of wind speed, wind angle and boat speed Raises a RegressorException - if - if - if",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.LeastSquareRegressor",
"url":7,
"doc":"A least square regressor based on scipy.optimize.curve_fit Parameters      model_func : function or callable The function which describes the model and is to be fitted. The function signature should be f(ws, wa,  params) -> bsp, where ws and wa are numpy.ndarrays resp. and params is a sequence of parameters that will be fitted init_vals : array_like ,optional Inital guesses for the optimal parameters of model_func that are passed to scipy.optimize.curve_fit Defaults to None"
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.LeastSquareRegressor.model_func",
"url":7,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.LeastSquareRegressor.optimal_params",
"url":7,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.LeastSquareRegressor.set_weights",
"url":7,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.regressor.LeastSquareRegressor.fit",
"url":7,
"doc":"Fits the model function to the given data, ie calculates the optimal parameters to minimize the sum of the squares of the residuals, see also  least squares   _ Parameters      data : array_like of shape (n, 3) Data to which the model function will be fitted, given as a sequence of points consisting of wind speed, wind angle and boat speed Raises a RegressorException if least-square minimization was not succesful, ie, if scipy.optimize.curve_fit raises a RuntimeError",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator",
"url":8,
"doc":"Contains the baseclass for Interpolators used in the TableExtension and PointcloudExtension class, that can also be used to create custom Interpolators. Also contains various predefined and usable interpolators"
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.scaled",
"url":8,
"doc":"dummy"
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.euclidean_norm",
"url":8,
"doc":"dummy"
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.InterpolatorException",
"url":8,
"doc":"Custom exception for errors that may appear whilst working with the Interpolator class and subclasses"
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.Interpolator",
"url":8,
"doc":"Base class for all Interpolator classes Abstract Methods         interpolate(self, w_pts)"
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.Interpolator.interpolate",
"url":8,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.IDWInterpolator",
"url":8,
"doc":"Basic inverse distance interpolator, based on the work of Shepard, \"A two-dimensional interpolation function for irregulary-spaced data\" For a given point grid_pt, that is to be interpolated, we calculate the distances d_pt =  grid-pt - pt[:2] for all considered measured points. Then we set the weights of a point pt to be w_pt = 1 / d_pt^p, for some nonnegative integer p The interpolated value on grid_pt then equals (\u03a3 w_pt pt[2]) / \u03a3 w_pt or if grid_pt is already a measured point pt , it will equal pt [2] Parameters      p : nonnegative int, optional Defaults to 2 norm : function or callable, optional Norm with which to calculate the distances, ie  . If nothing is passed, it will default to a scaled version of  . _2 Raises an InterpolatorException if the inputs are not of the specified type"
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.IDWInterpolator.interpolate",
"url":8,
"doc":"Interpolates a given grid_pt according to the above described method Parameters      w_pts : WeightedPoints Considered measured points grid_pt : numpy.ndarray of shape (2,) Point that is to be interpolated Returns    - out : int / float Interpolated values at grid_pt",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.ArithmeticMeanInterpolator",
"url":8,
"doc":"An Interpolator that gets the interpolated value according to the following procedure First the distance of the independent variables of all considered points and of the to interpolate point is calculated, ie  p[:2] - grid_pt  Then using a distribution, new weights are calculated based on the old weights, the previously calculated distances and other parameters depending on the distribution The value of the dependent variable of the interpolated point then equals s  (\u03a3 w_p  p) / \u03a3 w_p where s is an additional scaling factor In fact it is a more general approach to the inverse distance interpolator Parameters      s : positive int or float, optional Scaling factor for the arithmetic mean, Defaults to 1 norm : function or callable, optional Norm with which to calculate the distances, ie  . If nothing is passed, it will default to a scaled version of  . _2 distribution : function or callable, optional Function with which to calculate the updated weights. Should have the signature f(distances, old_weights,  parameters) -> new_weights If nothing is passed, it will default to gauss_potential, which calculated weights based on the formula \u03b2  exp(-\u03b1  old_weights  distances) params: Parameters to be passed to distribution Raises an InterpolatorException if the inputs are not of the specified type"
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.ArithmeticMeanInterpolator.interpolate",
"url":8,
"doc":"Interpolates a given grid_pt according to the above described method Parameters      w_pts : WeightedPoints Considered measured points grid_pt : numpy.ndarray of shape (2,) Point that is to be interpolated Returns    - out : int / float Interpolated values at grid_pt",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.gauss_potential",
"url":8,
"doc":"dummy"
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.ImprovedIDWInterpolator",
"url":8,
"doc":"An improved inverse distance interpolator, based on the work of Shepard, \"A two-dimensional interpolation function for irregulary-spaced data\" Should (only) be used together with the ScalingBall neighbourhood Instead of setting the weights as the normal inverse distance to some power, we set the weights in the following way: Let r be the radius of the ScalingBall with the center being some point grid_pt which is to be interpolated. For all considered measured points let d_pt be the same as in IDWInterpolator. If d_pt <= r/3 we set w_pt = 1 / d_pt. Otherwise we set w_pt = 27 / (4  r)  (d_pt / r - 1)^2 The resulting value on grid_pt will then be calculated the same way as in IDWInterpolator Parameters      norm : function or callable, optional Norm with which to calculate the distances, ie  . If nothing is passed, it will default to a scaled version of  . _2 Raises an InterpolatorException if the input is not of the specified type"
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.ImprovedIDWInterpolator.interpolate",
"url":8,
"doc":"Interpolates a given grid_pt according to the above described method Parameters      w_pts : WeightedPoints Considered measured points grid_pt : numpy.ndarray of shape (2,) Point that is to be interpolated Returns    - out : int / float Interpolated values at grid_pt Raises an InterpolatorException - if - if",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.ShepardInterpolator",
"url":8,
"doc":"A full featured inverse distance interpolator, based on the work of Shepard, \"A two-dimensional interpolation function for irregulary-spaced data\" Should (only) be used together with the ScalingBall neighbourhood Parameters      tol : positive float , optional Defautls to numpy.finfo(float).eps slope: positive float, optional Defaults to 0.1 norm : function or callable, optional Norm with which to calculate the distances, ie  . If nothing is passed, it will default to a scaled version of  . _2 Raises an InterpolatorException if inputs are not of the specified types"
},
{
"ref":"hrosailing.processing.pipelinecomponents.interpolator.ShepardInterpolator.interpolate",
"url":8,
"doc":"Interpolates a given grid_pt according to the above described method Parameters      w_pts : WeightedPoints Considered measured points grid_pt : numpy.ndarray of shape (2,) Point that is to be interpolated Returns    - out : int / float Interpolated values at grid_pt",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.filter",
"url":9,
"doc":"Contains the baseclass for Filters used in the PolarPipeline class, that can also be used to create custom Filters. Also contains two predefinied and usable filters, the QuantileFilter and the BoundFilter."
},
{
"ref":"hrosailing.processing.pipelinecomponents.filter.FilterException",
"url":9,
"doc":"Custom exception for errors that may appear whilst working with the Filter class and subclasses"
},
{
"ref":"hrosailing.processing.pipelinecomponents.filter.Filter",
"url":9,
"doc":"Base class for all filter classes Abstract Methods         filter(self, weights)"
},
{
"ref":"hrosailing.processing.pipelinecomponents.filter.Filter.filter",
"url":9,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.filter.QuantileFilter",
"url":9,
"doc":"A filter that filteres all points based on if their resp. weight lies above a certain quantile Parameters      percent: int or float, optional The quantile to be calculated Defaults to 25 Raises a FilterException, if percent is not in the interval [0, 100]"
},
{
"ref":"hrosailing.processing.pipelinecomponents.filter.QuantileFilter.filter",
"url":9,
"doc":"Filters a set of points given by their resp. weights according to the above described method Parameters      wts : numpy.ndarray of shape (n, ) Weights of the points that are to be filtered, given as a sequence of scalars Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing with points are filtered depending on their resp. weight",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.filter.BoundFilter",
"url":9,
"doc":"A filter that filters all points based on if their weight is outside an interval given by a lower and upper bound Parameters      upper_bound : int or float, optional The upper bound for the filter Defaults to 1 lower_bound : int or float, optional The lower bound for the filter Defaults to 0.5 Raises a FilterException if lower_bound is greater than upper_bound"
},
{
"ref":"hrosailing.processing.pipelinecomponents.filter.BoundFilter.filter",
"url":9,
"doc":"Filters a set of points given by their resp. weights according to the above described method Parameters      wts : numpy.ndarray of shape (n, ) Weights of the points that are to be filtered, given as a sequence of scalars Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing with points are filtered depending on their resp. weight",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.influencemodel",
"url":10,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.influencemodel.InfluenceException",
"url":10,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.influencemodel.InfluenceModel",
"url":10,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.influencemodel.InfluenceModel.remove_influence",
"url":10,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.influencemodel.InfluenceModel.add_influence",
"url":10,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.influencemodel.LinearCurrentModel",
"url":10,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.influencemodel.LinearCurrentModel.remove_influence",
"url":10,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.influencemodel.LinearCurrentModel.add_influence",
"url":10,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher",
"url":11,
"doc":"Contains the baseclass for Weighers used in the PolarPipeline class, that can also be used to create custom Weighers. Also contains two predefined and useable weighers, the CylindricMeanWeigher and the CylindricMemberWeigher, aswell as the WeightedPoints class, used to represent data points together with their respective weights"
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.scaled",
"url":11,
"doc":"dummy"
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.euclidean_norm",
"url":11,
"doc":"dummy"
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.WeightedPointsException",
"url":11,
"doc":"Custom exception for errors that may appear whilst working with the WeightedPoints class"
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.WeightedPoints",
"url":11,
"doc":"A class to weigh data points and represent them together with their respective weights Parameters      pts : array_like of shape (n, 3) Points that will be weight or paired with given weights wts : int, float or array_like of shape (n, ), optional If the weights of the points are known beforehand, they can be given as an argument. If weights are passed, they will be assigned to the points and no further weighing will take place If a scalar is passed, the points will all be assigned the same weight Defaults to None weigher : Weigher, optional Instance of a Weigher class, which will weigh the points Will only be used if weights is None If nothing is passed, it will default to CylindricMeanWeigher() tw : bool, optional Specifies if the given wind data should be viewed as true wind If False, wind data will be converted to true wind Defaults to True Raises a WeightedPointsException -"
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.WeightedPoints.__getitem__",
"url":11,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.WeightedPoints.points",
"url":11,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.WeightedPoints.weights",
"url":11,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.WeigherException",
"url":11,
"doc":"Custom exception for errors that may appear whilst working with the Weigher class and subclasses"
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.Weigher",
"url":11,
"doc":"Base class for all weigher classes Abstract Methods         weight(self, pts)"
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.Weigher.weigh",
"url":11,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.CylindricMeanWeigher",
"url":11,
"doc":"A weigher that weighs given points according to the following procedure: For a given point p and points pts we look at all the points pt in pts such that  pt[:d-1] - p[:d-1] <= r Then we take the mean m_p and standard deviation std_p of the dth component of all those points and set w_p = | m_p - p[d-1] | / std_p Parameters      radius : positive int or float, optional The radius of the considered cylinder, with infinite height, ie r Defaults to 1 norm : function or callable, optional Norm with which to evaluate the distances, ie  . If nothing is passed, it will default to  . _2 Raises a WeigherException if inputs are not of the specified types"
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.CylindricMeanWeigher.__repr__",
"url":11,
"doc":"Return repr(self).",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.CylindricMeanWeigher.weigh",
"url":11,
"doc":"Weigh given points according to the method described above Parameters      pts : numpy.ndarray of shape (n, 3) Points to be weight Returns    - wts : numpy.ndarray of shape (n, ) Normalized weights of the input points",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.CylindricMemberWeigher",
"url":11,
"doc":"A weigher that weighs given points according to the following procedure: For a given point p and points pts we look at all the points pt in pts such that |pt[0] - p[0]| <= h and  pt[1:] - p[1:] <= r Call the set of all such points P, then w_p =  P - 1 Parameters      radius : positive int or float, optional The radius of the considered cylinder, ie r Defaults to 1 length : nonnegative int of float, optional The height of the considered cylinder, ie h If length is 0, the cylinder is a d-1 dimensional ball Defaults to 1 norm : function or callable, optional Norm with which to evaluate the distances, ie  . If nothing is passed, it will default to  . _2 Raises a WeigherException if inputs are not of the specified types"
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.CylindricMemberWeigher.__repr__",
"url":11,
"doc":"Return repr(self).",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.weigher.CylindricMemberWeigher.weigh",
"url":11,
"doc":"Weigh given points according to the method described above Parameters      pts : numpy.ndarray of shape (n, 3) Points to be weight Returns    - wts : numpy.ndarray of shape (n, ) Normalized weights of the input points",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood",
"url":12,
"doc":"Contains the baseclass for Neighbourhoods used in the TableExtension and PointcloudExtension class, that can also be used to create custom Neighbourhoods. Also contains various predefined and usable neighbourhoods"
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.scaled",
"url":12,
"doc":"dummy"
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.euclidean_norm",
"url":12,
"doc":"dummy"
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.NeighbourhoodException",
"url":12,
"doc":"Custom exception for errors that may appear whilst working with the Neighbourhood class and subclasses"
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.Neighbourhood",
"url":12,
"doc":"Base class for all neighbourhood classes Abstract Methods         is_contained_in(self, pts)"
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.Neighbourhood.is_contained_in",
"url":12,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.Ball",
"url":12,
"doc":"A class to describe a closed 2-dimensional ball centered around the origin, ie { x in R^2 :  x <= r } Parameters      norm : function or callable, optional The norm for which the ball is described, ie  . If nothing is passed, it will default to a scaled version of  . _2 radius : positive int or float, optional The radius of the ball, ie r Defaults to 1 Raises a NeighbourhoodException if inputs are not of the specified types"
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.Ball.is_contained_in",
"url":12,
"doc":"Checks given points for membership. Parameters      pts : array_like of shape (n, 2) Points that will be checked for membership Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing which of the input points is a member of the neighbourhood Raises a NeighbourhoodException if the input is not of the specified type",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.ScalingBall",
"url":12,
"doc":"A class to represent a closed 2-dimensional ball centered around the origin, ie { x in R^2 :  x <= r }, where the radius r will be dynamically determined, such that there are always a certain amount of given points contained in the ball Parameters      min_pts : positive int The minimal amount of certain given points that should be contained in the scaling ball max_pts : positive int The \"maximal\" amount of certain given points that should be contained in the scaling ball. Mostly used for initial guess of a \"good\" radius. Also to guarantee that on average, the scaling ball will contain (min_pts + max_pts) / 2 points of certain given points It is also unlikely that the scaling ball will contain more than max_pts points norm : function or callable, optional The norm for which the scaling ball is described, ie  . If nothing is passed, it will default to a scaled version of  . _2 Raises a NeighbourhoodException - if inputs are not of the specified types - if max_pts is smaller or equal to min_pts"
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.ScalingBall.is_contained_in",
"url":12,
"doc":"Checks given points for membership, and scales ball so that at least min_pts points are contained in it Parameters      pts : array_like of shape (n, 2) Points that will be checked for membership Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing which of the input points is a member of the neighbourhood Raises a NeighbourhoodException if the input is not of the specified type",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.Ellipsoid",
"url":12,
"doc":"A class to represent a closed d-dimensional ellipsoid centered around the origin, ie T(B), where T is an invertible linear transformation, and B is a closed d-dimensional ball, centered around the origin. It will be represented using the equivalent formulation: { x in R^2 :  T^-1 x <= r } Parameters      lin_trans: array_like of shape (2,2), optional The linear transformation which transforms the ball into the given ellipsoid, ie T If nothing is passed, it will default to I_2, the 2x2 unit matrix, ie the ellipsoid will be a ball norm : function or callable, optional The norm for which the ellipsoid is described, ie  . If nothing is passed, it will default to a scaled version of  . _2 radius : positive int or float, optional The radius of the ellipsoid, ie r Defaults to 1 Raises a NeighbourhoodException - if the inputs are not of the specified types - if lin_trans is not invertible"
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.Ellipsoid.is_contained_in",
"url":12,
"doc":"Checks given points for membership. Parameters      pts : array_like of shape (n, 2) Points that will be checked for membership Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing which of the input points is a member of the neighbourhood Raises a NeighbourhoodException if the input is not of the specified type",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.Cuboid",
"url":12,
"doc":"A class to represent a d-dimensional closed cuboid, ie { x in R^2 : |x_i| <= b_i, i=1,2 } Parameters      norm : function or callable, optional The 1-d norm used to measure the length of the x_i, ie |.| If nothing is passed, it will default to the absolute value |.| dimensions: tuple of length 2, optional The 'length' of the 'sides' of the cuboid, ie the b_i If nothing is passed, it will default to (1,1) Raises a NeighbourhoodException if inputs are not of the specified types"
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.Cuboid.is_contained_in",
"url":12,
"doc":"Checks given points for membership. Parameters      pts : array_like of shape (n, 2) Points that will be checked for membership Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing which of the input points is a member of the neighbourhood Raises a NeighbourhoodException if the input is not of the specified type",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.Polytope",
"url":12,
"doc":"A class to represent a general 2-dimensional polytope, ie the convex hull P = conv(x_1,  ., x_n) of some n points x_1 , ., x_n or equivalent as the (bounded) intersection of m half spaces P = { x in R^2 : Ax <= b } Parameters      mat: array_like of shape (m, 2), optional matrix to represent the normal vectors a_i of the half spaces, ie A = (a_1,  . , a_m)^t If nothing is passed, it will default to (I_2, -I_2)^t, where I_d is the d-dimensional unit matrix b: array_like of shape (m, ), optional vector to represent the  . b_i of the half spaces, ie b = (b_1,  . , b_m)^t If nothing is passed, it will default to (1, .,1) Raises a NeighbourhoodException - if inputs are not of the specified types - mat or b contain NaN or infinite values Warning    - Does not check wether the polytope given by mat and b is a polytope, ie if P is actually bounded"
},
{
"ref":"hrosailing.processing.pipelinecomponents.neighbourhood.Polytope.is_contained_in",
"url":12,
"doc":"Checks given points for membership. Parameters      pts : array_like of shape (n, 2) Points that will be checked for membership Returns    - mask : numpy.ndarray of shape (n, ) Boolean array describing which of the input points is a member of the neighbourhood Raises a NeighbourhoodException if the input is not of the specified type",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.datahandler",
"url":13,
"doc":""
},
{
"ref":"hrosailing.processing.pipelinecomponents.datahandler.HandlerException",
"url":13,
"doc":"Custom exception for errors that may appear whilst working with the DataHandler class and subclasses"
},
{
"ref":"hrosailing.processing.pipelinecomponents.datahandler.DataHandler",
"url":13,
"doc":"Base class for all datahandler classes Abstract Methods         handle(self, data)"
},
{
"ref":"hrosailing.processing.pipelinecomponents.datahandler.DataHandler.handle",
"url":13,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.datahandler.ArrayHandler",
"url":13,
"doc":"A data handler to handle data, given as an array_like sequence. Doesn't really do anything since error handling and array conversion is handeled by the pipeline itself Only needed for general layout of the pipeline"
},
{
"ref":"hrosailing.processing.pipelinecomponents.datahandler.ArrayHandler.handle",
"url":13,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.datahandler.CsvFileHandler",
"url":13,
"doc":"A data handler to extract data from a .csv file with the first three columns representing wind speed, wind angle, and boat speed respectively"
},
{
"ref":"hrosailing.processing.pipelinecomponents.datahandler.CsvFileHandler.handle",
"url":13,
"doc":"Reads a .csv file and extracts the contained data points The delimiter used in the .csv file Parameters      data : path-like Path to a .csv file pandas_kw : Returns    - out : pandas.Dataframe Raises a HandlerException",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.datahandler.NMEAFileHandler",
"url":13,
"doc":"A data handler to extract data from a text file containing certain nmea sentences Parameters     - mode : string, optional In the case where there is more recorded wind data than speed data, specifies how to handle the surplus - \"interpolate\": handles the surplus by taking convex combinations of two recorded speed datas together with the recorded wind data \"between\" those two points to create multiple data points - \"mean\": handles the surplus by taking the mean of the wind data \"belonging\" to a given speed data to create a singe data point Defaults to \"interpolate\" Raises a HandlerException if"
},
{
"ref":"hrosailing.processing.pipelinecomponents.datahandler.NMEAFileHandler.handle",
"url":13,
"doc":"Reads a text file containing nmea-sentences and extracts data points based on recorded wind speed, wind angle, and speed over water Function looks for sentences of type: - MWV for wind data - VHW for speed trough water Parameters      data : path-like Path to a text file, containing nmea-0183 sentences Returns    - out : list of lists of length 3 Raises a HandlerException - if  data doesn't contain relevant nmea senteces - if nmea senteces are not sorted - if an error occurs whilst reading - if an error occurs whilst parsing of the nmea senteces - if an error occurs during conversion of apperant wind",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler",
"url":14,
"doc":"Contains the baseclass for Samplers used in the PointcloudExtension class, that can also be used to create custom Samplers. Also contains various predefined and usable samplers."
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler.SamplerException",
"url":14,
"doc":"Custom exception for errors that may appear whilst working with the Sampler class and subclasses"
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler.Sampler",
"url":14,
"doc":"Base class for all sampler classes Abstract Methods         sample(self, pts)"
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler.Sampler.sample",
"url":14,
"doc":"",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler.UniformRandomSampler",
"url":14,
"doc":"A sampler that produces a number of uniformly distributed samples, which all lie in the convex hull of certain given points Parameters      n_samples : positive int Amount of samples that will be produced by the sampler Raises SamplerException if input is not of specified type"
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler.UniformRandomSampler.sample",
"url":14,
"doc":"Produces samples according to the above described procedure Parameters      pts : array_like of shape (n, 2) Points in whose convex hull the produced samples will lie Returns    - samples : numpy.ndarray of shape (n_samples, 2) samples produced by the above described method",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler.FibonacciSampler",
"url":14,
"doc":"A sampler that produces sample points on a moved and scaled version of the spiral (sqrt(x) cos(x), sqrt(x) sin(x , such that the angles are distributed equidistantly by the inverse golden ratio. The sample points all lie in the smallest enclosing circle of given data points. Inspired by \u00c1lvaro Gonzz\u00e1lez - \"Measurement of areas on a sphere using Fibonacci and latitude\u2013longitude lattices\" Parameters      n_samples : positive int Amount of samples that will be produced by the sampler Raises SamplerException if input is not of specified type"
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler.FibonacciSampler.sample",
"url":14,
"doc":"Produces samples according to the above described procedure Parameters      pts : array_like of shape (n, 2) Points in whose convex hull the produced samples will lie Returns    - samples : numpy.ndarray of shape (n_samples, 2) samples produced by the above described method",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler.ArchimedianSampler",
"url":14,
"doc":"A sampler that produces a number of approximately equidistant sample points on a moved and scaled version of the archimedean spiral (x cos(x), x sin(x . The sample points all lie in the smallest enclosing circle of given data points. Inspired by https: agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007GC001581 Parameters      n_samples : positive int Amount of samples that will be produced by the sampler Raises SamplerException if input is not of specified type"
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler.ArchimedianSampler.sample",
"url":14,
"doc":"Produces samples according to the above described procedure Parameters      pts : array_like of shape (n, 2) Points in whose convex hull the produced samples will lie Returns    - samples : numpy.ndarray of shape (n_samples, 2) samples produced by the above described method",
"func":1
},
{
"ref":"hrosailing.processing.pipelinecomponents.sampler.make_circle",
"url":14,
"doc":"dummy"
}
]