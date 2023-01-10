# pylint: disable=missing-module-docstring

import csv
import itertools
import warnings
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np

from ._basepolardiagram import (
    PolarDiagram,
    PolarDiagramException,
    PolarDiagramInitializationException,
)
from ._plotting import _get_new_axis, plot_convex_hull_multisails
from ._polardiagramtable import PolarDiagramTable


class NotYetImplementedWarning(Warning):
    """Simple warning for not fully finished implementations."""


class PolarDiagramMultiSails(PolarDiagram):
    """A class to represent, visualize and work with
    a polar diagram made up of multiple sets of sails,
    represented by a `PolarDiagramTable`.


    Class methods aren't fully developed yet. Take care when
    using this class.

    Parameters
    ----------
    pds : Sequence of PolarDiagramTable objects
        Polar diagrams belonging to different sets of sails,
        given as tables, that share the same wind speeds.

    sails : Sequence, optional
        Custom names for the sails. Length should be equal to `pds`.
        If it is not equal it will either be cut off at the appropriate
        length or will be addended with `"Sail i"` to the appropriate length.

        Only important for the legend of plots or the `to_csv()`-method.

        If nothing is passed, the names will be `"Sail i"`, i = 0...n-1,
        where `len(pds) = n`.

    Raises
    ------
    PolarDiagramInitializationException
        If the polar tables don't share the same wind speeds.
    """

    def __init__(self, pds, sails=None):
        warnings.warn(
            (
                "class features aren't all fully developed yet and/or might "
                "change behaviour heavily in the future. "
                "Take care when using this class"
            ),
            category=NotYetImplementedWarning,
        )
        ws = pds[0].wind_speeds
        for pd in pds:
            if not np.array_equal(ws, pd.wind_speeds):
                raise PolarDiagramInitializationException(
                    "wind speed resolution of `pds` does not coincide"
                )

        if sails is None:
            sails = [f"Sail {i}" for i in range(len(pds))]
        elif len(sails) < len(pds):
            sails = list(sails) + [
                f"Sail {i}" for i in range(len(sails) + 1, len(pds))
            ]
        elif len(sails) > len(pds):
            sails = list(sails)
            sails = sails[: len(pds)]

        self._sails = list(sails)
        self._tables = list(pds)

    @property
    def sails(self):
        return self._sails

    @property
    def wind_speeds(self):
        return self._tables[0].wind_speeds

    @property
    def tables(self):
        return self._tables

    def __getitem__(self, item) -> PolarDiagramTable:
        """"""
        try:
            index = self.sails.index(item)
        except ValueError as ve:
            raise PolarDiagramException(
                "`item` is not a name of a sail"
            ) from ve

        return self.tables[index]

    def __str__(self):
        tables = [str(pd) for pd in self._tables]
        names = [str(sail) for sail in self._sails]
        out = []
        for name, table in zip(names, tables):
            out.append(name)
            out.append("\n")
            out.append(table)
            out.append("\n\n")

        return "".join(out)

    def __repr__(self):
        return f"PolarDiagramMultiSails({self.tables}, {self.sails})"

    def to_csv(self, csv_path):
        """Creates a .csv file with delimiter ',' and the
        following format:

            `PolarDiagramMultiSails`
            TWS
            `self.wind_speeds`
            [Sail
            TWA
            `table.wind_angles`
            Boat speeds
            `table.boat_speeds`]

        Parameters
        ----------
        csv_path : path_like
            Path to a .csv file or where a new .csv file will be created.
        """
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file, delimiter=",")
            csv_writer.writerow([self.__class__.__name__])
            csv_writer.writerow(["TWS"])
            csv_writer.writerow(self.wind_speeds)
            for sail, table in zip(self.sails, self.tables):
                csv_writer.writerow([sail])
                csv_writer.writerow(["TWA"])
                csv_writer.writerow(table.wind_angles)
                csv_writer.writerow(["BSP"])
                csv_writer.writerows(table.boat_speeds)

    @classmethod
    def __from_csv__(cls, file):
        csv_reader = csv.reader(file, delimiter=",")
        next(csv_reader)
        ws_resolution = [literal_eval(ws) for ws in next(csv_reader)]
        sails, pds = _extract_polardiagrams(csv_reader, ws_resolution)
        return PolarDiagramMultiSails(pds, sails)

    def symmetrize(self):
        """Constructs a symmetric version of the polar diagram, by
        mirroring each `PolarDiagramTable` at the 0° - 180° axis and
        returning a new instance. See also the `symmetrize()`-method
        of the `PolarDiagramTable` class.

        Warning
        -------
        Should only be used if all the wind angles of the `PolarDiagramTables`
        are each on one side of the 0° - 180° axis, otherwise this can lead
        to duplicate data, which can overwrite or live alongside old data.
        """
        pds = [pd.symmetrize() for pd in self._tables]
        return PolarDiagramMultiSails(pds, self._sails)

    def get_slices(self, ws):
        """For given wind speeds, return the slices of the polar diagram
        corresponding to them.

        The slices are equal to the corresponding
        columns of the table together with `self.wind_angles`.

        Parameters
        ----------
        ws : tuple of length 2, iterable, int or float, optional
            Slices of the polar diagram table, given as either:

            - a tuple of length 2 specifying an interval of considered
            wind speeds,
            - an iterable containing only elements of `self.wind_speeds`,
            - a single element of `self.wind_speeds`.

            If nothing it passed, it will default to `self.wind_speeds`.


        Returns
        -------
        ws : list
            The wind speeds corresponding to the slices.

        wa : list of numpy.ndarray
            A list of the corresponding wind angles for each slice.

        bsp : list of numpy.ndarray
            `bsp[i][j]` contains the resulting boat speed for wind speed
            `ws[i]` and wind angle `wa[i][j]`.

        members : list of str
            `members[j]` contains the name of the sail corresponding to the
            wind `ws[i]` and `wa[i][j]` for any value of `i`.
        """
        wa = []
        temp = []
        for pd in self._tables:
            ws, w, b = pd.get_slices(ws)
            wa.append(w)
            temp.append(b)

        flatten = itertools.chain.from_iterable
        members = [[self._sails[i]] * len(w) for i, w in enumerate(wa)]
        members = list(flatten(members))

        wa = [np.asarray(wa).ravel()] * len(ws)
        bsp = []
        for i in range(len(ws)):
            b = np.asarray([b_[:, i] for b_ in temp]).ravel()
            bsp.append(b)

        return ws, wa, bsp, members

    def plot_polar(
        self,
        ws=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a polar plot of one or more slices of the polar diagram.

        Parameters
        ----------
        ws : tuple of length 2, iterable, int or float, optional
            Slices of the polar diagram table, given as either:

            - a tuple of length 2 specifying an interval of considered
            wind speeds,
            - an iterable containing only elements of `self.wind_speeds`,
            - a single element of `self.wind_speeds`.

            The slices are then equal to the corresponding
            columns of the table together with `self.wind_angles`.

            If nothing it passed, it will default to `self.wind_speeds`.

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

        colors : color_like or sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices.

            - If a color_like is passed, all slices will be plotted in the
            respective color.
            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed.
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds.
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a sequence of `(ws, color)` pairs.

            Defaults to `("green", "red")`.

        show_legend : bool, optional
            Specifies whether a legend will be shown next to the plot.

            The type of legend depends on the color options.

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend`.

            Defaults to `False`.

        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.

            - If 2 colors are passed, a colorbar will be created.
            In this case see `matplotlib.colorbar.Colorbar` for possible
            keywords and their effect.
            - Otherwise, a legend will be created.
            In this case see `matplotlib.legend.Legend` for possible keywords
            and their effect.

            Will only be used if `show_legend` is `True`.

        plot_kw : Keyword arguments
            Keyword arguments to change various appearances of the plot.

            See `matplotlib.axes.Axes.plot` for possible keywords and their
            effects.

        Raises
        ------
        PolarDiagramException
            If at least one element of `ws` is not in `self.wind_speeds`.
        PolarDiagramException
            If the given interval doesn't contain any slices of the
            polar diagram.
        """
        if ax is None:
            ax = plt.axes(projection="polar")

        for i, pd in enumerate(self._tables):
            if i == 0 and show_legend:
                pd.plot_polar(
                    ws, ax, colors, show_legend, legend_kw, **plot_kw
                )
                continue

            pd.plot_polar(ws, ax, colors, False, None, **plot_kw)

    def plot_flat(
        self,
        ws=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a cartesian plot of one or more slices of the polar diagram.

        Parameters
        ----------
        ws : tuple of length 2, iterable, int or float, optional
            Slices of the polar diagram table, given as either:

            - a tuple of length 2 specifying an interval of considered
            wind speeds,
            - an iterable containing only elements of `self.wind_speeds`,
            - a single element of `self.wind_speeds`.

            The slices are then equal to the corresponding
            columns of the table together with `self.wind_angles`.

            If nothing is passed, it will default to `self.wind_speeds`.

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

        colors : color_like or sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices.

            - If a color_like is passed, all slices will be plotted in the
            respective color.
            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed.
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds.
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a sequence of `(ws, color)` pairs.

            Defaults to `("green", "red")`.

        show_legend : bool, optional
            Specifies whether or not a legend will be shown next to the plot.

            The type of legend depends on the color options.

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend`.

            Defaults to `False`.

        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.

            - If 2 colors are passed, a colorbar will be created.
            In this case see `matplotlib.colorbar.Colorbar` for possible
            keywords and their effect.
            - Otherwise, a legend will be created.
            In this case see `matplotlib.legend.Legend` for possible keywords
            and their effect.

            Will only be used if `show_legend` is `True`.

        plot_kw : Keyword arguments
            Keyword arguments to change various appearances of the plot.

            See `matplotlib.axes.Axes.plot` for possible keywords and their
            effects.

        Raises
        ------
        PolarDiagramException
            If at least one element of `ws` is not in `self.wind_speeds`.
        PolarDiagramException
            If the given interval doesn't contain any slices of the
            polar diagram.
        """
        if ax is None:
            ax = _get_new_axis("rectilinear")

        for i, pd in enumerate(self._tables):
            if i == 0 and show_legend:
                pd.plot_flat(ws, ax, colors, show_legend, legend_kw, **plot_kw)
                continue

            pd.plot_flat(ws, ax, colors, False, None, **plot_kw)

    def plot_3d(self, ax=None, colors=("green", "red")):
        """Creates a 3d plot of the polar diagram.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes instance where the plot will be created.

        colors: tuple of two (2) color_likes, optional
            Color pair determining the color gradient with which the
            polar diagram will be plotted.

            Will be determined by the corresponding wind speeds.

            Defaults to `("green", "red")`.
        """
        if ax is None:
            ax = plt.axes(projection="3d")

        for pd in self._tables:
            pd.plot_3d(ax, colors)

    def plot_color_gradient(
        self,
        ax=None,
        colors=("green", "red"),
        marker=None,
        ms=None,
        show_legend=False,
        **legend_kw,
    ):
        """

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

        colors : tuple of two (2) color_likes, optional
            Color pair determining the color gradient with which the
            polar diagram will be plotted.

            Will be determined by the corresponding boat speed.

            Defaults to `("green", "red")`.

        marker : matplotlib.markers.Markerstyle or equivalent, optional
            Markerstyle for the created scatter plot.

            Defaults to `"o"`.

        ms : float or array_like of fitting shape, optional
            Marker size in points**2.

        show_legend : bool, optional
            Specifies whether or not a legend will be shown next
            to the plot.

            Legend will be a `matplotlib.colorbar.Colorbar` instance.

            Defaults to `False`.

        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.

            - If 2 colors are passed, a colorbar will be created.
            In this case see `matplotlib.colorbar.Colorbar` for possible
            keywords and their effect.
            - Otherwise, a legend will be created.
            In this case see `matplotlib.legend.Legend` for possible keywords
            and their effect.

            Will only be used if `show_legend` is `True`.

        """
        warnings.warn(
            "feature isn't implemented yet", category=NotYetImplementedWarning
        )

    def plot_convex_hull(
        self,
        ws=None,
        ax=None,
        colors=None,
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Computes the (separate) convex hull of one or more
        slices of the polar diagram and creates a polar plot of them.

        Parameters
        ----------
        ws : tuple of length 2, iterable, int or float, optional
            Slices of the polar diagram table, given as either:

            - a tuple of length 2 specifying an interval of considered
            wind speeds,
            - an iterable containing only elements of `self.wind_speeds`,
            - a single element of `self.wind_speeds`.

            The slices are then equal to the corresponding
            columns of the table together with `self.wind_angles`.

            If nothing it passed, it will default to `self.wind_speeds`.

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

        colors : subscriptable iterable of color_likes, optional

        show_legend : bool, optional
            Specifies whether a legend will be shown next to the plot.

            The type of legend depends on the color options.

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend`.

            Defaults to `False`.

        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.

            - If 2 colors are passed, a colorbar will be created.
            In this case see `matplotlib.colorbar.Colorbar` for possible
            keywords and their effect.
            - Otherwise, a legend will be created.
            In this case see `matplotlib.legend.Legend` for possible keywords
            and their effect.

            Will only be used if `show_legend` is `True`.

        plot_kw : Keyword arguments
            Keyword arguments to change various appearances of the plot.

            See `matplotlib.axes.Axes.plot` for possible keywords and their
            effects.

        Raises
        ------
        PolarDiagramException
            If at least one element of `ws` is not in `self.wind_speeds`.
        """
        ws, wa, bsp, members = self.get_slices(ws)
        plot_convex_hull_multisails(
            ws, wa, bsp, members, ax, colors, show_legend, legend_kw, **plot_kw
        )


def _extract_polardiagrams(csv_reader, ws_resolution):
    sails = []
    pds = []

    while True:
        try:
            sails.append(next(csv_reader)[0])
            next(csv_reader)
            wa_resolution = [literal_eval(wa) for wa in next(csv_reader)]
            next(csv_reader)
            bsps = [
                [literal_eval(bsp) for bsp in row]
                for row in itertools.islice(csv_reader, len(wa_resolution))
            ]
            pds.append(PolarDiagramTable(ws_resolution, wa_resolution, bsps))
        except StopIteration:
            break

    return sails, pds
