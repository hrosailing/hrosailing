# pylint: disable=missing-module-docstring
# pylint: disable=import-outside-toplevel

from abc import ABC, abstractmethod


class PolarDiagramException(Exception):
    """Exception raised if some nonstandard error occurs,
    while doing something with polar diagrams
    """


class PolarDiagramInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a PolarDiagram
    """


class PolarDiagram(ABC):
    """Base class for all polar diagrams

    Abstract Methods
    ----------------
    to_csv(csv_path)

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

    def pickling(self, pkl_path):
        """Writes PolarDiagram instance to a .pkl file

        Since the pickle module can't guarantee security, but
        we have found no other way to serialize toplevel functions
        in python, we have decided to omit a depickling function
        and leave that up to the user.

        Parameters
        ----------
        pkl_path: path-like
            Path to a .pkl file or where a new .pkl file will be created
        """
        with open(pkl_path, "wb") as file:
            import pickle

            pickle.dump(self, file)

    @abstractmethod
    def to_csv(self, csv_path):
        """This method should, given a path, write a .csv file in
        the location, containing human readable information about the
        polar diagram object that called the method
        """

    @classmethod
    def __from_csv__(cls, csv_reader):
        raise NotImplementedError(f"hro-format for {cls} not implemented")

    @abstractmethod
    def symmetrize(self):
        """This method should return a new PolarDiagram object that is a
        symmetric (i.e. mirrored along the 0 - 180° axis) version of the
        polar diagram object that called the method
        """

    @abstractmethod
    def get_slices(self, ws):
        """This method should, given a number of wind speeds, return
        a list of the given wind speeds as well as wind angles and
        corresponding boat speeds, that reflect how the vessel behaves at
        the given wind speeds
        """

    def plot_polar_slice(self, ws, ax=None, **plot_kw):
        """Creates a polar plot of a given slice of the
        polar diagram

        Parameters
        ----------
        ws : int/float
            Slice of the polar diagram

            For a description of what the slice is made of,
            see the plot_polar()-method of the respective
            PolarDiagram subclasses

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created

            If nothing is passed, the function will create
            a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot
        """
        self.plot_polar(
            ws, ax, colors=None, show_legend=False, legend_kw=None, **plot_kw
        )

    def plot_flat_slice(self, ws, ax=None, **plot_kw):
        """Creates a cartesian plot of a given slice of the
        polar diagram

        Parameters
        ----------
        ws : int/float
            Slice of the polar diagram

            For a description of what the slice is made of,
            see the plot_flat()-method of the respective
            PolarDiagram subclass

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot
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
        corresponding to `ws`, of the polar diagram object
        """

    @abstractmethod
    def plot_3d(self):
        """This method should create a 3d plot of the polar diagram object"""

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
        to the corresponding boat speeds
        """

    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        """Computes the convex hull of a given slice of
        the polar diagram and creates a polar plot of it

        Parameters
        ----------
        ws : int/float
            Slice of the polar diagram

            For a description of what the slice is made of,
            see the plot_convex_hull()-method of the respective
            PolarDiagram subclass

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot
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
        a polar plot of them
        """
