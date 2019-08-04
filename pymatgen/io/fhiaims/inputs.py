# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import os
import re
import itertools
import warnings
import logging
import math
import glob
import subprocess

import numpy as np

from numpy.linalg import det
from collections import OrderedDict, namedtuple
from hashlib import md5

from monty.io import zopen
from monty.os.path import zpath
from monty.json import MontyDecoder
from monty.os import cd

from enum import Enum
from tabulate import tabulate

import scipy.constants as const

from pymatgen import SETTINGS, __version__
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.periodic_table import Element, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.string import str_delimited
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.typing import PathLike
from monty.json import MSONable

"""
Classes for reading/manipulating/writing VASP input files. All major VASP input
files.
"""

__author__ = "Shyue Ping Ong, Geoffroy Hautier, Rickard Armiento, " + \
             "Vincent L Chevrier, Stephen Dacek"
__copyright__ = "Copyright 2011, The Materials Project"
__version__ = "1.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyuep@gmail.com"
__status__ = "Production"
__date__ = "Jul 16, 2012"


logger = logging.getLogger(__name__)


class Control(MSONable):
    """
    Object for representing the data in a POSCAR or CONTCAR file.
    Please note that this current implementation. Most attributes can be set
    directly.

    Args:
        structure (Structure):  Structure object.
        comment (str): Optional comment line for POSCAR. Defaults to unit
            cell formula of structure. Defaults to None.
        selective_dynamics (Nx3 array): bool values for selective dynamics,
            where N is number of sites. Defaults to None.
        true_names (bool): Set to False is the names in the POSCAR are not
            well-defined and ambiguous. This situation arises commonly in
            vasp < 5 where the POSCAR sometimes does not contain element
            symbols. Defaults to True.
        velocities (Nx3 array): Velocities for the POSCAR. Typically parsed
            in MD runs or can be used to initialize velocities.
        predictor_corrector (Nx3 array): Predictor corrector for the POSCAR.
            Typically parsed in MD runs.

    .. attribute:: structure

        Associated Structure.

    .. attribute:: comment

        Optional comment string.

    .. attribute:: true_names

        Boolean indication whether Poscar contains actual real names parsed
        from either a POTCAR or the POSCAR itself.

    .. attribute:: selective_dynamics

        Selective dynamics attribute for each site if available. A Nx3 array of
        booleans.

    .. attribute:: velocities

        Velocities for each site (typically read in from a CONTCAR). A Nx3
        array of floats.

    .. attribute:: predictor_corrector

        Predictor corrector coordinates and derivatives for each site; i.e.
        a list of three 1x3 arrays for each site (typically read in from a MD 
        CONTCAR).

    .. attribute:: predictor_corrector_preamble

        Predictor corrector preamble contains the predictor-corrector key,
        POTIM, and thermostat parameters that precede the site-specic predictor 
        corrector data in MD CONTCAR

    .. attribute:: temperature

        Temperature of velocity Maxwell-Boltzmann initialization. Initialized
        to -1 (MB hasn"t been performed).
    """

    def __init__(self, structure, comment=None, selective_dynamics=None,
                 true_names=True, velocities=None, predictor_corrector=None,
                 predictor_corrector_preamble=None):
        if structure.is_ordered:
            site_properties = {}
            if selective_dynamics:
                site_properties["selective_dynamics"] = selective_dynamics
            if velocities:
                site_properties["velocities"] = velocities
            if predictor_corrector:
                site_properties["predictor_corrector"] = predictor_corrector
            structure = Structure.from_sites(structure)
            self.structure = structure.copy(site_properties=site_properties)
            self.true_names = true_names
            self.comment = structure.formula if comment is None else comment
            self.predictor_corrector_preamble = predictor_corrector_preamble
        else:
            raise ValueError("Structure with partial occupancies cannot be "
                             "converted into POSCAR!")

        self.temperature = -1

    @property
    def velocities(self):
        return self.structure.site_properties.get("velocities")

    @property
    def selective_dynamics(self):
        return self.structure.site_properties.get("selective_dynamics")

    @property
    def predictor_corrector(self):
        return self.structure.site_properties.get("predictor_corrector")

    @velocities.setter
    def velocities(self, velocities):
        self.structure.add_site_property("velocities", velocities)

    @selective_dynamics.setter
    def selective_dynamics(self, selective_dynamics):
        self.structure.add_site_property("selective_dynamics",
                                         selective_dynamics)

    @predictor_corrector.setter
    def predictor_corrector(self, predictor_corrector):
        self.structure.add_site_property("predictor_corrector",
                                         predictor_corrector)

    @property
    def site_symbols(self):
        """
        Sequence of symbols associated with the Poscar. Similar to 6th line in
        vasp 5+ POSCAR.
        """
        syms = [site.specie.symbol for site in self.structure]
        return [a[0] for a in itertools.groupby(syms)]

    @property
    def natoms(self):
        """
        Sequence of number of sites of each type associated with the Poscar.
        Similar to 7th line in vasp 5+ POSCAR or the 6th line in vasp 4 POSCAR.
        """
        syms = [site.specie.symbol for site in self.structure]
        return [len(tuple(a[1])) for a in itertools.groupby(syms)]

    def __setattr__(self, name, value):
        if name in ("selective_dynamics", "velocities"):
            if value is not None and len(value) > 0:
                value = np.array(value)
                dim = value.shape
                if dim[1] != 3 or dim[0] != len(self.structure):
                    raise ValueError(name + " array must be same length as" +
                                     " the structure.")
                value = value.tolist()
        super().__setattr__(name, value)


    def from_file(self, filename, read_velocities=True):
        """
        Reads a Poscar from a file.

        The code will try its best to determine the elements in the POSCAR in
        the following order:
        1. If check_for_POTCAR is True, the code will try to check if a POTCAR
        is in the same directory as the POSCAR and use elements from that by
        default. (This is the VASP default sequence of priority).
        2. If the input file is Vasp5-like and contains element symbols in the
        6th line, the code will use that if check_for_POTCAR is False or there
        is no POTCAR found.
        3. Failing (2), the code will check if a symbol is provided at the end
        of each coordinate.

        If all else fails, the code will just assign the first n elements in
        increasing atomic number, where n is the number of species, to the
        Poscar. For example, H, He, Li, ....  This will ensure at least a
        unique element is assigned to each site and any analysis that does not
        require specific elemental properties should work fine.

        Args:
            filename (str): File name containing Poscar data.
            check_for_POTCAR (bool): Whether to check if a POTCAR is present
                in the same directory as the POSCAR. Defaults to True.
            read_velocities (bool): Whether to read or not velocities if they
                are present in the POSCAR. Default is True.

        Returns:
            string
        """
        with open(filename, 'rt') as f:
            content = f.readlines()

        return self.from_string(''.join(content))

    @staticmethod
    def from_string(data, default_names=None, read_velocities=True):
        """
        Reads a Poscar from a string.

        The code will try its best to determine the elements in the POSCAR in
        the following order:
        1. If default_names are supplied and valid, it will use those. Usually,
        default names comes from an external source, such as a POTCAR in the
        same directory.
        2. If there are no valid default names but the input file is Vasp5-like
        and contains element symbols in the 6th line, the code will use that.
        3. Failing (2), the code will check if a symbol is provided at the end
        of each coordinate.

        If all else fails, the code will just assign the first n elements in
        increasing atomic number, where n is the number of species, to the
        Poscar. For example, H, He, Li, ....  This will ensure at least a
        unique element is assigned to each site and any analysis that does not
        require specific elemental properties should work fine.

        Args:
            data (str): String containing Poscar data.
            default_names ([str]): Default symbols for the POSCAR file,
                usually coming from a POTCAR in the same directory.
            read_velocities (bool): Whether to read or not velocities if they
                are present in the POSCAR. Default is True.

        Returns:
            Poscar object.
        """
        # "^\s*$" doesn't match lines with no whitespace
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


        lattice = Lattice(lat) if is_periodic else np.array([[150,0,0],[0,150,0],[0,0,150]])
        struct = Structure(lattice=lattice, species=[Element(e) for e in name],
                           coords=[np.array(v) for v in atom], validate_proximity=True,
                           to_unit_cell=False, coords_are_cartesian=not is_fractional)
        return struct


    def get_string(self, significant_figures=6):
        """
        Returns a string to be written as geometry.in

        Args:
            significant_figures (int): No. of significant figures to
                output all quantities. Defaults to 6. Note that positions are
                output in fixed point, while velocities are output in
                scientific format.

        Returns:
            String representation of geometry.in
        """

        is_periodic = False if self.structure.lattice == Lattice([[150,0,0],[0,150,0],[0,0,150]]) else True

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
        """
        String representation of Poscar file.
        """
        return self.get_string()

    def write_file(self, filename, **kwargs):
        """
        Writes POSCAR to a file. The supported kwargs are the same as those for
        the Poscar.get_string method and are passed through directly.
        """
        with zopen(filename, "wt") as f:
            f.write(self.get_string(**kwargs))

    def as_dict(self):
        return {"@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "structure": self.structure.as_dict(),
                "true_names": self.true_names,
                "selective_dynamics": np.array(
                    self.selective_dynamics).tolist(),
                "velocities": self.velocities,
                "predictor_corrector": self.predictor_corrector,
                "comment": self.comment}

    @classmethod
    def from_dict(cls, d):
        return Poscar(Structure.from_dict(d["structure"]),
                      comment=d["comment"],
                      selective_dynamics=d["selective_dynamics"],
                      true_names=d["true_names"],
                      velocities=d.get("velocities", None),
                      predictor_corrector=d.get("predictor_corrector", None))
