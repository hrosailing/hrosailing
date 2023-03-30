# pylint: disable=missing-module-docstring

import csv
import itertools
from ast import literal_eval

import numpy as np

from ._basepolardiagram import PolarDiagram
from ._polardiagramtable import PolarDiagramTable
from ._reading import from_csv


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

    Attributes
    ---------
    sails (property) : list of str
        A read only version of the list of sails.

    diagrams (property) : list of `PolarDiagram`
        The different polar diagrams.

    SeeAlso
    -------
    `PolarDiagram`
    """

    def __init__(self, pds, sails=None):
        if sails is None:
            sails = [f"Sail {i}" for i in range(len(pds))]
        elif len(sails) < len(pds):
            sails = list(sails) + [
                f"Sail {i}" for i in range(len(sails), len(pds))
            ]
        elif len(sails) > len(pds):
            sails = list(sails)
            sails = sails[: len(pds)]

        self._sails = list(sails)
        self._diagrams = list(pds)

    @property
    def sails(self):
        return self._sails

    @property
    def diagrams(self):
        return self._diagrams

    def __getitem__(self, item) -> PolarDiagramTable:
        """"""
        try:
            index = self.sails.index(item)
        except ValueError as ve:
            raise ValueError("`item` is not a name of a sail") from ve

        return self._diagrams[index]

    def __str__(self):
        tables = [str(pd) for pd in self._diagrams]
        names = [str(sail) for sail in self._sails]
        out = []
        for name, table in zip(names, tables):
            out.append(name)
            out.append("\n")
            out.append(table)
            out.append("\n\n")

        return "".join(out)

    def __repr__(self):
        return f"PolarDiagramMultiSails({[repr(diagram) for diagram in self._diagrams]}, {self.sails})"

    def __call__(self, ws, wa):
        return max(diagram(ws, wa) for diagram in self._diagrams)

    @property
    def default_points(self):
        return np.row_stack([pd.default_points for pd in self._diagrams])

    def get_slices(self, ws=None, n_steps=None, full_info=False, **kwargs):
        """
        Parameters
        ---------
        **kwargs :
            The keyword arguments are forwarded to the `ws_to_slices` methods
            of the polardiagrams in `self.diagrams`.

        Returns
        --------
        slices : list of slices
            Concatenations of the slices of the polardiagrams in
            `self.diagrams`.

        See also
        -------
        `PolarDiagram.get_slices`
        """
        return super().get_slices(ws, n_steps, full_info, **kwargs)

    def ws_to_slices(self, ws, **kwargs):
        """
        See also
        -------
        `PolarDiagramMultisails.get_slices`
        `PolarDiagram.ws_to_slices`
        """
        all_slices = [pd.ws_to_slices(ws, **kwargs) for pd in self._diagrams]
        slices = [
            np.column_stack(slice_collection)
            for slice_collection in zip(*all_slices)
        ]
        return slices

    def get_slice_info(self, ws, slices, **kwargs):
        """
        Returns
        --------
        info : list of lists
            Contains the sail name used for each record. Organized in the
            same manner as the slices.

        See also
        ----------
        `PolarDiagram.get_slices`
        """
        all_slices = [pd.ws_to_slices(ws, **kwargs) for pd in self._diagrams]
        slice_infos = [
            [
                sail
                for sail, slice in zip(self.sails, slice_collection)
                for _ in range(slice.shape[1])
            ]
            for slice_collection in zip(*all_slices)
        ]
        return slice_infos

    def to_csv(self, csv_path):
        """Creates a .csv file with delimiter ',' and the
        following format:

            `PolarDiagramMultiSails
            Sails:.....`

            The sub-diagrams are automatically saved in the files
            `{csv_path}_sail_{i}.csv`

        Parameters
        ----------
        csv_path : path_like
            Path to a .csv file or where a new .csv file will be created.
        """
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file, delimiter=":")
            csv_writer.writerow([self.__class__.__name__])
            csv_writer.writerow(["Sails"]+self.sails)

        for i, diagram in enumerate(self._diagrams):
            sub_csv_path = self._get_sub_csv_path(csv_path, i)
            diagram.to_csv(sub_csv_path)

    @staticmethod
    def _get_sub_csv_path(path, i):
        path =  "_".join([path.split(".")[0], "sail", str(i)])
        return f"{path}.csv"

    @classmethod
    def __from_csv__(cls, file):
        path = file.name
        csv_reader = csv.reader(file, delimiter=":")
        sails = next(csv_reader)[1:]
        diagrams = []
        for i, sail in enumerate(sails):
            sub_path = PolarDiagramMultiSails._get_sub_csv_path(path, i)
            diagrams.append(
                from_csv(sub_path)
            )
        return PolarDiagramMultiSails(diagrams, sails)

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
        pds = [pd.symmetrize() for pd in self.diagrams]
        return PolarDiagramMultiSails(pds, self._sails)

    @property
    def default_slices(self):
        all_defaults = np.concatenate(
            [pd.default_slices for pd in self._diagrams]
        )
        return np.linspace(all_defaults.min(), all_defaults.max(), 20)

