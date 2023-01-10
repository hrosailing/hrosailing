# pylint: disable=missing-module-docstring

from abc import ABC, abstractmethod


class PolarDiagramException(Exception):
    """Exception raised if some nonstandard error occurs,
    while doing something with polar diagrams.
    """


class PolarDiagramInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a `PolarDiagram`.
    """


class PolarDiagram(ABC):
    """Base class for all polar diagrams.

    Abstract Methods
    ----------------
    to_csv(csv_path)

    __from_csv__(cls, file)

    symmetrize()

    get_slices(ws)

    plot_polar(
        ws,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw
    )

    plot_flat(
        ws,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw
    )

    plot_3d(ax=None, **plot_kw)

    plot_color_gradient(
        ax=None,
        colors=("green", "red"),
        marker=None,
        ms=None,
        show_legend=False,
        **legend_kw,
    )

    plot_convex_hull(
        ws,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
    """

    @abstractmethod
    def to_csv(self, csv_path):
        """This method should, given a path, write a .csv file in
        the location, containing human-readable information about the
        polar diagram object that called the method.

        Parameters
        ----------
        csv_path: str
            Path of the created csv file.
        """

    @classmethod
    @abstractmethod
    def __from_csv__(cls, file):
        """"""

    @abstractmethod
    def symmetrize(self):
        """This method should return a new `PolarDiagram` object that is a
        symmetric (i.e. mirrored along the 0 - 180° axis) version of the
        `PolarDiagram` object called the method.
        """

    @abstractmethod
    def get_slices(self, ws):
        """This method should, given a number of wind speeds, return
        a list of the given wind speeds as well as wind angles and
        corresponding boat speeds, that reflect how the vessel behaves at
        the given wind speeds.

        Parameters
        ----------
        ws : int/float
            Description of slices of the polar diagram to be plotted.

            For a description of what the slice is made of,
            see the `get_slices()`-method of the respective
            `PolarDiagram` subclass.
        """

    def plot_polar_slice(self, ws, ax=None, **plot_kw):
        """Creates a polar plot of a given slice of the
        polar diagram.

        Parameters
        ----------
        ws : int/float
            Description of slices of the polar diagram to be plotted.

            For a description of what the slice is made of,
            see the `plot_polar()`-method of the respective
            `PolarDiagram` subclass.

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes.

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            `matplotlib.axes.Axes.plot` function, to change
            certain appearances of the plot.
        """
        self.plot_polar(
            ws, ax, colors=None, show_legend=False, legend_kw=None, **plot_kw
        )

    def plot_flat_slice(self, ws, ax=None, **plot_kw):
        """Creates a cartesian plot of a given slice of the
        polar diagram.

        Parameters
        ----------
        ws : int/float
            Slice of the polar diagram.

            For a description of what the slice is made of,
            see the `plot_flat()`-method of the respective
            `PolarDiagram` subclass.

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes.

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            `matplotlib.axes.Axes.plot` function, to change
            certain appearances of the plot.
        """
        self.plot_flat(
            ws, ax, colors=None, show_legend=False, legend_kw=None, **plot_kw
        )

    @abstractmethod
    def plot_polar(
        self,
        ws=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """This method should create a polar plot of one or more slices,
        corresponding to `ws`, of the polar diagram object.

        Parameters
        ---------

        ws : int/float
            Description of slices of the polar diagram to be plotted.

            For details refer to the `plot_polar()`-method of the respective
            `PolarDiagram` subclass.

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

        colors : sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices.

            For details refer to the `plot_polar()`-method of the respective
            `PolarDiagram` subclass.

            Defaults to `("green", "red")`.

        show_legend : bool, optional
            Specifies whether or not a legend will be shown next to the plot.

            For details refer to the `plot_polar()`-method of the respective
            `PolarDiagram` subclass.

            Defaults to `False`.

        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the legend.

            See `matplotlib.colorbar.Colorbar` for possible keywords
            when a colorbar is created and `matplotlib.legend.Legend` for
            possible keywords in cases where a Legend is created.

            For details refer to the `plot_polar()`-method of the respective
            `PolarDiagram` subclass.

            Will only be used if `show_legend` is `True`.

        plot_kw : Keyword arguments
            Keyword arguments to change various appearances of the plot.

            See `matplotlib.axes.Axes.plot` for possible keywords and their
            effects.
        """

    @abstractmethod
    def plot_flat(
        self,
        ws=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """This method should create a cartesian plot of one or more slices,
        corresponding to `ws`, of the `PolarDiagram` object.


        Parameters
        ----------
        ws : tuple of length 2, iterable, int or float, optional
            Slices of the polar diagram.

            For details refer to the `plot_flat()`-method of the respective
            `PolarDiagram` subclass.

            If nothing is passed, it will default to `(0, 20)`.

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

        colors : sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices.

            For details refer to the `plot_flat()`-method of the respective
            `PolarDiagram` subclass.

            Defaults to `("green", "red")`.

        show_legend : bool, optional
            Specifies whether a legend will be shown next to the plot.

            For details refer to the `plot_flat()`-method of the respective
            `PolarDiagram` subclass.

            Defaults to `False`.

        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the legend.

            See `matplotlib.colorbar.Colorbar` and `matplotlib.legend.Legend` for
            possible keywords and their effects.

            Will only be used if `show_legend` is `True`.

        plot_kw : Keyword arguments
            Keyword arguments to change various appearances of the plot.

            See `matplotlib.axes.Axes.plot` for possible keywords and their
            effects.
        """

    @abstractmethod
    def plot_3d(self):
        """This method should create a 3d plot of the polar diagram object."""

    @abstractmethod
    def plot_color_gradient(
        self,
        ax=None,
        colors=("green", "red"),
        marker=None,
        ms=None,
        show_legend=False,
        **legend_kw,
    ):
        """This method should create 'wind speed vs. wind angle'
        color gradient plot of the polar diagram object with respect
        to the corresponding boat speeds.

        Parameters
        ----------

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

        colors : tuple of two (2) color_likes, optional
            Color pair determining the color gradient with which the
            polar diagram will be plotted.

            For details refer to the `plot_color_gradient()`-method of
            the respective `PolarDiagram` subclass.

            Defaults to `("green", "red")`.

        marker : matplotlib.markers.Markerstyle or equivalent, optional
            Markerstyle for the created scatter plot.

            Defaults to `"o"`

        ms : float or array_like of fitting shape, optional
            Marker size in points**2.

        show_legend : bool, optional
            Specifies whether or not a legend will be shown next
            to the plot.

            Legend will be a `matplotlib.colorbar.Colorbar` instance.

            Defaults to `False`.

        legend_kw : Keyword arguments
            Keyword arguments to change position and appearance of the legend.

            See `matplotlib.legend.Legend` for possible keywords and
            their effects.

            Will only be used if `show_legend` is `True`.
        """

    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        """Computes the convex hull of a given slice of
        the polar diagram and creates a polar plot of it.

        Parameters
        ----------
        ws : int/float
            Slice of the polar diagram.

            For a description of what the slice is made of,
            see the `plot_convex_hull()`-method of the respective
            `PolarDiagram` subclass.

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes.

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            `matplotlib.axes.Axes.plot` function, to change
            certain appearances of the plot.
        """
        self.plot_convex_hull(
            ws, ax, colors=None, show_legend=False, legend_kw=None, **plot_kw
        )

    @abstractmethod
    def plot_convex_hull(
        self,
        ws=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """This method should compute the convex hull of one or multiple
        slices, corresponding to `ws`, of the polar diagram and then create
        a polar plot of them.

        Parameters
        ----------
        ws : tuple of length 2, iterable, int or float, optional
            Slices of the polar diagram.

            For details refer to the `plot_convex_hull()`-method of
            the respective `PolarDiagram` subclass.

            If nothing is passed, it will default to `(0, 20)`.

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

        colors : sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices.

            For details refer to the `plot_convex_hull()`-method of
            the respective `PolarDiagram` subclass.

            Defaults to `("green", "red")`.

        show_legend : bool, optional
            Specifies whether or not a legend will be shown next to the plot.

            The type of legend depends on the color options.

            For details refer to the `plot_convex_hull()`-method of
            the respective `PolarDiagram` subclass.

            Defaults to `False`.

        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the legend.

            See `matplotlib.colorbar.Colorbar` and `matplotlib.legend.Legend`
            for possible keywords and their effects.

            Will only be used if show_legend is `True`.

        plot_kw : Keyword arguments
            Keyword arguments to change various appearances of the plot.

            See `matplotlib.axes.Axes.plot` for possible keywords and their
            effects.

        """
