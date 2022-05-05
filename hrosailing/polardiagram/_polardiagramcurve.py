# pylint: disable=missing-module-docstring

import csv
from ast import literal_eval
from inspect import getmembers, isfunction

import numpy as np

import hrosailing.pipelinecomponents.modelfunctions as model

from ._basepolardiagram import (PolarDiagram, PolarDiagramException,
                                PolarDiagramInitializationException)
from ._plotting import (plot_color_gradient, plot_convex_hull, plot_flat,
                        plot_polar, plot_surface)

MODEL_FUNCTIONS = dict(getmembers(model, isfunction))


class PolarDiagramCurve(PolarDiagram):
    """A class to represent, visualize and work with a polar diagram
    given by a fitted curve/surface.

    Parameters
    ----------
    f : function
        Curve/surface that describes the polar diagram, given as
        a function, with the signature `f(ws, wa, *params) -> bsp`,
        where `ws`, `wa` and should be `array_like` of shape `(n,)`

    params : Sequence
        Optimal parameters for `f`

    radians : bool, optional
        Specifies if `f` takes the wind angles in radians or degrees

        Defaults to `False`

    Raises
    ------
    PolarDiagramInitializationException
        If `f` is not callable

        If `params` contains not enough parameters for `f`
    """

    def __init__(self, f, *params, radians=False):
        if not callable(f):
            raise PolarDiagramInitializationException("`f` is not callable")

        if not self._check_enough_params(f, params):
            raise PolarDiagramInitializationException(
                "`params` is an incorrect amount of parameters for `f`"
            )

        self._f = f
        self._params = params
        self._rad = radians

    @staticmethod
    def _check_enough_params(func, params):
        try:
            func(1, 1, *params)
            return True
        except (IndexError, TypeError):
            return False

    def __repr__(self):
        return (
            f"PolarDiagramCurve(f={self._f.__name__},"
            f"{self._params}, radians={self._rad})"
        )

    def __call__(self, ws, wa):
        return self.curve(ws, wa, *self.parameters)

    @property
    def curve(self):
        """Returns a read only version of `self._f`"""
        return self._f

    @property
    def parameters(self):
        """Returns a read only version of `self._params`"""
        return self._params

    @property
    def radians(self):
        """Returns a read only version of `self._rad`"""
        return self._rad

    def to_csv(self, csv_path):
        """Creates a .csv file with delimiter ':' and the
        following format:

            PolarDiagramCurve
            Function: `self.curve.__name__`
            Radians: `self.radians`
            Parameters: `self.parameters`

        Parameters
        ----------
        csv_path : path-like
            Path to a .csv file or where a new .csv file will be created

        Raises
        ------
        OSError
            If no write permission is given for file
        """
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file, delimiter=":")
            csv_writer.writerow([self.__class__.__name__])
            csv_writer.writerow(["Function"] + [self.curve.__name__])
            csv_writer.writerow(["Radians"] + [str(self.radians)])
            csv_writer.writerow(["Parameters"] + list(self.parameters))

    @classmethod
    def __from_csv__(cls, file):
        csv_reader = csv.reader(file, delimiter=":")
        function = next(csv_reader)[1]
        radians = literal_eval(next(csv_reader)[1])
        params = [literal_eval(value) for value in next(csv_reader)[1:]]

        if function not in MODEL_FUNCTIONS:
            raise PolarDiagramInitializationException(
                f"No valid function, named {function}"
            )

        function = MODEL_FUNCTIONS[function]
        return PolarDiagramCurve(function, *params, radians=radians)

    def symmetrize(self):
        """Constructs a symmetric version of the polar diagram,
        by mirroring it at the 0° - 180° axis and returning a new instance
        """

        def sym_func(ws, wa, *params):
            return 0.5 * (
                self._f(ws, wa, *params) + self._f(ws, 360 - wa, *params)
            )

        return PolarDiagramCurve(
            sym_func, *self.parameters, radians=self.radians
        )

    def get_slices(self, ws=None, stepsize=None):
        """For given wind speeds, return the slices of the polar diagram
        corresponding to them

        Slices are equal to `self(w, wa)` where `w` goes through
        the given values in `ws` and `wa` goes through a fixed
        number of angles between 0° and 360°

        Parameters
        ----------
        ws : tuple of length 2, iterable, int or float, optional
            Slices of the polar diagram given as either

            - a tuple of length 2, specifying an interval of considered
            wind speeds. The amount of slices taken from that interval are
            determined by the parameter `stepsize`
            - an iterable of specific wind speeds
            - a single wind speed

            If nothing is passed, it will default to `(0, 20)`

        stepsize : positive int or float, optional
            Specfies the amount of slices taken from the given
            wind speed interval

            Will only be used if `ws` is a tuple of length 2

            If nothing is passed, it will default to `ws[1] - ws[0]`

        Returns
        -------
        slices : tuple
            Slices of the polar diagram, given as a tuple of length 3,
            consisting of the given wind speeds `ws`, `self.wind_angles`
            (in rad) and a list of arrays containing the
            corresponding boat speeds

        Raises
        ------
        PolarDiagramException
            If `stepsize` is nonpositive
        """
        if ws is None:
            ws = (0, 20)

        if isinstance(ws, (int, float)):
            ws = [ws]
        elif isinstance(ws, tuple) and len(ws) == 2:
            if stepsize is None:
                stepsize = int(round(ws[1] - ws[0]))

            if stepsize <= 0:
                raise PolarDiagramException("`stepsize` is nonpositive")

            ws = list(np.linspace(ws[0], ws[1], stepsize))

        wa = np.linspace(0, 360, 1000)
        if self.radians:
            wa = np.deg2rad(wa)

        bsp = [self(np.array([w] * 1000), wa) for w in ws]

        if not self.radians:
            wa = np.deg2rad(wa)

        return ws, wa, bsp

    # pylint: disable=arguments-renamed
    def plot_polar(
        self,
        ws=None,
        stepsize=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a polar plot of one or more slices of the polar diagram

        Parameters
        ----------
        ws : tuple of length 2, iterable, int or float, optional
            Slices of the polar diagram given as either

            - a tuple of length 2, specifying an interval of considered
            wind speeds. The amount of slices taken from that interval are
            determined by the parameter `stepsize`
            - an iterable of specific wind speeds
            - a single wind speed

            Slices will then equal `self(w, wa)` where `w` goes through
            the given values in `ws` and `wa` goes through a fixed
            number of angles between 0° and 360°

            If nothing is passed, it will default to `(0, 20)`

        stepsize : positive int or float, optional
            Specfies the amount of slices taken from the given
            wind speed interval

            Will only be used if `ws` is a tuple of length 2

            If nothing is passed, it will default to `ws[1] - ws[0]`

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

        colors : sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices

            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a `(ws, color)` pair

            Defaults to `("green", "red")`

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next to the plot

            The type of legend depends on the color options

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend` instance

            Defaults to `False`

        legend_kw : dict, optional
            Keyword arguments to change position and appearence of the legend

            See matplotlib.colorbar.Colorbar and matplotlib.legend.Legend for
            possible keywords and their effects

            Will only be used if show_legend is `True`

        plot_kw : Keyword arguments
            Keyword arguments to change various appearences of the plot

            See matplotlib.axes.Axes.plot for possible keywords and their
            effects
        """
        ws, wa, bsp = self.get_slices(ws, stepsize)
        wa = [wa] * len(ws)

        plot_polar(
            ws,
            wa,
            bsp,
            ax,
            colors,
            show_legend,
            legend_kw,
            _lines=True,
            **plot_kw,
        )

    # pylint: disable=arguments-renamed
    def plot_flat(
        self,
        ws=None,
        stepsize=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a cartesian plot of one or more slices of the polar diagram

        Parameters
        ----------
        ws : tuple of length 2, iterable, int or float, optional
            Slices of the polar diagram given as either

            - a tuple of length 2, specifying an interval of considered
            wind speeds. The amount of slices taken from that interval are
            determined by the parameter `stepsize`
            - an iterable of specific wind speeds
            - a single wind speed

            Slices will then equal `self(w, wa)` where `w` goes through
            the given values in `ws` and `wa` goes through a fixed
            number of angles between 0° and 360°

            If nothing is passed, it will default to (0, 20)

        stepsize : positive int or float, optional
            Specfies the amount of slices taken from the given
            wind speed interval

            Will only be used if `ws` is a tuple of length 2

            If nothing is passed, it will default to `ws[1] - ws[0]`

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

        colors : sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices

            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a `(ws, color)` pair

            Defaults to `("green", "red")`

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next to the plot

            The type of legend depends on the color options

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend` instance

            Defaults to `False`

        legend_kw : dict, optional
            Keyword arguments to change position and appearence of the legend

            See matplotlib.colorbar.Colorbar and matplotlib.legend.Legend for
            possible keywords and their effects

            Will only be used if show_legend is `True`

        plot_kw : Keyword arguments
            Keyword arguments to change various appearences of the plot

            See matplotlib.axes.Axes.plot for possible keywords and their
            effects
        """
        ws, wa, bsp = self.get_slices(ws, stepsize)
        wa = [np.rad2deg(wa)] * len(ws)

        plot_flat(
            ws,
            wa,
            bsp,
            ax,
            colors,
            show_legend,
            legend_kw,
            _lines=True,
            **plot_kw,
        )

    def plot_3d(
        self, ws=None, stepsize=None, ax=None, colors=("green", "red")
    ):
        """Creates a 3d plot of a part of the polar diagram

        Parameters
        ----------
        ws : tuple of length 2, optional
            A region of the polar diagram given as an interval of
            wind speeds

            Slices will then equal `self(w, wa)` where `w` goes through
            the given values in `ws` and `wa` goes through a fixed
            number of angles between 0° and 360°

            If nothing is passed, it will default to `(0, 20)`

        stepsize : positive int or float, optional
            Specfies the amount of slices taken from the given
            interval in `ws`

            If nothing is passed, it will default to `100`

        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes instance where the plot will be created.

        colors: tuple of two (2) color_likes, optional
            Color pair determining the color gradient with which the
            polar diagram will be plotted

            Will be determined by the corresponding wind speeds

            Defaults to `("green", "red")`
        """
        if stepsize is None:
            stepsize = 100

        ws, wa, bsp = self.get_slices(ws, stepsize)
        bsp = np.array(bsp).T
        ws, wa = np.meshgrid(ws, wa)
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)

        plot_surface(ws, wa, bsp, ax, colors)

    def plot_color_gradient(
        self,
        ws=None,
        stepsize=None,
        ax=None,
        colors=("green", "red"),
        marker=None,
        ms=None,
        show_legend=False,
        **legend_kw,
    ):
        """Creates a 'wind speed vs. wind angle' color gradient plot
        of a part of the polar diagram with respect to the corresponding
        boat speeds

        Parameters
        ----------
        ws :  tuple of length 3, optional
            A region of the polar diagram given as an interval of
            wind speeds

            Slices will then equal `self(w, wa)` where `w` goes through
            the given values in `ws` and `wa` goes through a fixed
            number of angles between 0° and 360°

            If nothing is passed, it will default to `(0, 20)`

        stepsize : positive int or float, optional
            Specfies the amount of slices taken from the given
            interval in `ws`

            If nothing is passed, it will default to `100`

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

        colors : tuple of two (2) color_likes, optional
            Color pair determining the color gradient with which the
            polar diagram will be plotted

            Will be determined by the corresponding boat speed

            Defaults to `("green", "red")`

        marker : matplotlib.markers.Markerstyle or equivalent, optional
            Markerstyle for the created scatter plot

            Defaults to `"o"`

        ms : float or array_like of fitting shape, optional
            Marker size in points**2

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next
            to the plot

            Legend will be a `matplotlib.colorbar.Colorbar` instance

            Defaults to `False`

        legend_kw : Keyword arguments
            Keyword arguments to change position and appearence of the legend

            See matplotlib.legend.Legend for possible keywords and
            their effects

            Will only be used if show_legend is `True`
        """
        if stepsize is None:
            stepsize = 100

        ws, wa, bsp = self.get_slices(ws, stepsize)
        wa = np.rad2deg(wa)
        ws, wa = np.meshgrid(ws, wa)
        bsp = np.array(bsp).T

        plot_color_gradient(
            ws.ravel(),
            wa.ravel(),
            bsp.ravel(),
            ax,
            colors,
            marker,
            ms,
            show_legend,
            **legend_kw,
        )

    # pylint: disable=arguments-renamed
    def plot_convex_hull(
        self,
        ws=None,
        stepsize=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Computes the (seperate) convex hull of one or more
        slices of the polar diagram and creates a polar plot of them

        Parameters
        ----------
        ws : tuple of length 2, iterable, int or float, optional
            Slices of the polar diagram given as either

            - a tuple of length 2, specifying an interval of considered
            wind speeds. The amount of slices taken from that interval are
            determined by the parameter `stepsize`
            - an iterable of specific wind speeds
            - a single wind speed

            Slices will then equal `self(w, wa)` where `w` goes through
            the given values in `ws` and `wa` goes through a fixed
            number of angles between 0° and 360°

            If nothing is passed, it will default to `(0, 20)`

        stepsize : positive int or float, optional
            Specfies the amount of slices taken from the given
            wind speed interval

            Will only be used if `ws` is a tuple of length 2

            If nothing is passed, it will default to `ws[1] - ws[0]`

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be create

        colors : sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices

            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a `(ws, color)` pair

            Defaults to `("green", "red")`

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next to the plot

            The type of legend depends on the color options

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend`

            Defaults to `False`

        legend_kw : dict, optional
            Keyword arguments to change position and appearence of the legend

            See matplotlib.colorbar.Colorbar and matplotlib.legend.Legend for
            possible keywords and their effects

            Will only be used if show_legend is `True`

        plot_kw : Keyword arguments
            Keyword arguments to change various appearences of the plot

            See matplotlib.axes.Axes.plot for possible keywords and their
            effects
        """
        ws, wa, bsp = self.get_slices(ws, stepsize)
        wa = [wa] * len(ws)

        plot_convex_hull(
            ws,
            wa,
            bsp,
            ax,
            colors,
            show_legend,
            legend_kw,
            _lines=True,
            **plot_kw,
        )
