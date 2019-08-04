# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import re
import logging

import numpy as np

from monty.io import zopen

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from monty.json import MSONable

"""
Class for FHI-aims minimal IO   
"""

__author__ = "Shyue Ping Ong, Geoffroy Hautier, Rickard Armiento, " + \
             "Vincent L Chevrier, Stephen Dacek, Jan Kloppenburg"
__copyright__ = "Copyright 2011, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Jan Kloppenburg"
__email__ = "jank@numphys.org"
__status__ = "Beta"
__date__ = "5Aug2019"


logger = logging.getLogger(__name__)


class Control(MSONable):
    """
    Object for representing the data in a geometry.in file.

    Args:
        structure (Structure):  Structure object.
        selective_dynamics (Nx3 array): bool values for selective dynamics,
            where N is number of sites. Defaults to None.

    .. attribute:: structure

        Associated Structure.

    .. attribute:: selective_dynamics

        Selective dynamics attribute for each site if available. A Nx3 array of
        booleans.
    """

    def __init__(self, structure, selective_dynamics=None):
        if structure.is_ordered:
            site_properties = {}
            if selective_dynamics:
                site_properties["selective_dynamics"] = selective_dynamics
            structure = Structure.from_sites(structure)
            self.structure = structure.copy(site_properties=site_properties)
        else:
            raise ValueError("Structure with partial occupancies cannot be "
                             "converted into Structure object!")

    @property
    def selective_dynamics(self):
        return self.structure.site_properties.get("selective_dynamics")

    @selective_dynamics.setter
    def selective_dynamics(self, selective_dynamics):
        self.structure.add_site_property("selective_dynamics",
                                         selective_dynamics)

    def from_file(self, filename):
        """"""
        with open(filename, 'rt') as f:
            content = f.readlines()
        return self.from_string(''.join(content))

    @staticmethod
    def from_string(data):
        """
        Reads structure from string.

        Args:
            data (str): String containing geometry data.

        Returns:
            Structure object.
        """

        chunks = re.split(r"\n\s*\n", data.rstrip(), flags=re.MULTILINE)
        try:
            if chunks[0] == "":
                chunks.pop(0)
                chunks[0] = "\n" + chunks[0]
        except IndexError:
            raise ValueError("Empty File")

        chunks = chunks[0].split('\n')

        is_periodic = False
        is_fractional = False

        lat = []
        atom = []
        name = []
        for ch in chunks:
            if 'lattice_vector' in ch:
                is_periodic = True
                lat.append([float(x) for x in ch.split()[1:4]])
            if 'atom_frac' in ch:
                is_fractional = True
                atom.append([float(x) for x in ch.split()[1:4]])
                name.append(ch.split()[4])
            if 'atom' in ch:
                is_fractional = False
                atom.append([float(x) for x in ch.split()[1:4]])
                name.append(ch.split()[4])

        lattice = Lattice(lat) if is_periodic else np.array([[150, 0, 0],
                                                             [0, 150, 0],
                                                             [0, 0, 150]])
        return Structure(lattice=lattice, species=[Element(e) for e in name],
                         coords=[np.array(v) for v in atom], validate_proximity=True,
                         to_unit_cell=False, coords_are_cartesian=not is_fractional)

    def get_string(self):
        """
        Returns:
            String representation of geometry.in
        """
        is_periodic = False if self.structure.lattice == Lattice([[150, 0, 0],
                                                                  [0, 150, 0],
                                                                  [0, 0, 150]]) else True
        out = []
        if is_periodic:
            for l in self.structure.lattice.matrix:
                out.append('lattice_vector {:6.6f} {:6.6f} {:6.6f}'.format(l[0], l[1], l[2]))
        for c, n in zip(self.structure.cart_coords, self.structure.species):
            out.append('atom {:6.6f} {:6.6f} {:6.6f} {}'.format(c[0], c[1], c[2], n))

        return '\n'.join(out)

    def __repr__(self):
        return self.get_string()

    def __str__(self):
        return self.get_string()

    def write_file(self, filename):
        """
        Writes geometry to file.
        """
        with zopen(filename, "wt") as f:
            f.write(self.get_string())

    def as_dict(self):
        return {"@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "structure": self.structure.as_dict(),
                "selective_dynamics": np.array(self.selective_dynamics).tolist()}

    @classmethod
    def from_dict(cls, d):
        return Structure.from_dict(d)
