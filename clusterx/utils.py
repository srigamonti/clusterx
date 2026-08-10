# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from __future__ import annotations

import copy
import os
import time
import warnings
from contextlib import contextmanager
from typing import List, Optional, Union
import random

import numpy as np
from ase.cell import Cell
from ase import Atoms
from ase.build.supercells import (  # needed by make_supercell
    clean_matrix,
    lattice_points_in_supercell,
)
from ase.data import chemical_symbols as cs


class RunningAverage:
    def __init__(self, n: int):
        self.n = n
        self.average: float
        self.values = []

    def update(self, value) -> None:
        """Update the window of values by appending `value`, and compute new average."""
        self.values.append(value)
        if len(self.values) > self.n:
            value_0 = self.values.pop(0)
            self.average += (value - value_0) / self.n
        elif len(self.values) == 1:
            self.average = value
        else:
            diff = (value - self.average) / len(self.values)
            self.average += diff

    def get_average(self) -> float:
        return self.average

    def update_and_get_average(self, value) -> float:
        self.update(value)
        return self.average


class ListDict(object):
    """
    Datastructure that allows efficient (O(1)) insert, remove and random sampling.
    Adapted from https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice.
    """

    def __init__(self):
        self.item_to_position = {}
        self.items = []

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def add(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choice(self, rng: random.Random = random):
        return rng.choice(self.items)


def transformation_containing_sphere_diag(r: float, cell: np.ndarray | Cell, full_sphere: bool = False) -> np.ndarray:
    """Return diagonal transformation p of `cell` that describe a supercell,
    such that a all pairs of points with distance `r` are contained in the
    supercell. Only valid for dim=3 (len(cell.shape)==3).

    If `full_sphere`, returned transformation is large enough,
    such that a sphere with radius `r` can be fully contained in corresponding
    supercell, with the center at any point in the original cell."""
    c = np.zeros((3, 3))
    h = np.zeros(3)
    # Get distances h between parallel planes of the unit cell
    for i in range(3):
        c[i] = np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])
        h[i] = np.dot(cell[i], c[i] / np.linalg.norm(c[i]))
    # number of repetitions of unit cell along each lattice vector to contain largest cluster
    if full_sphere:
        return np.diag([int(p) for p in np.ceil(2 * r / h + 1)])
    else:
        return np.diag([int(p) for p in np.ceil(r / h)])


def take_tile(grid: np.ndarray, i_grid: np.ndarray, p_reduced: tuple):
    """Map values from grid to reduced grid around grid index i_grid.

    **parameters**
    ``grid``: ndarray, shape (pX, pY, pZ, ..., pd n)
    ``i_grid``: ndarray, shape (d + 1,)
    ``p_reduced``: tuple of int, shape (d,)
    """
    for axis, (d, p) in enumerate(zip(i_grid[:-1], p_reduced)):
        grid = np.take(grid, np.arange(d - p // 2, d + p // 2 + 1), axis=axis, mode="wrap")
    i_grid_new = np.array((np.array(p_reduced) // 2).tolist() + [i_grid[-1]])
    return grid, i_grid_new


def is_diagonal(x: np.ndarray):
    if len(np.shape(x)) == 1:
        return True
    return np.count_nonzero(x - np.diag(np.diagonal(x))) == 0


def lattice_wrap_index(index, lattice_shape):
    """Returns periodic lattice index for a given iterable index.
    Inspired by: https://stackoverflow.com/questions/38066785/np-ndarray-with-periodic-boundary-conditions

    **parameters**
    ``index``: iterable
        one integer for each axis
    ``lattice_shape``: tuple
        the shape of the lattice to index to
    """
    if not hasattr(index, "__iter__"):
        raise ValueError(f"wrap index does not support integer index: {index}")
    if len(index) != len(lattice_shape):
        raise ValueError(f"index {index} has incompatible shape with {lattice_shape}")
    if any(isinstance(i, slice) for i in index):
        raise ValueError("wrap index does not suppport slices")
    if len(index) == len(lattice_shape):
        mod_index = tuple(((i % s + s) % s for i, s in zip(index, lattice_shape)))
        return mod_index
    raise ValueError(f"Unexpected index: {index}, given shape {lattice_shape}")


@contextmanager
def _timed(label):
    print(f"[{label}] ...starting")  # print immediately
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"[{label}] completed in {end - start:.4f} seconds")


# @contextmanager
# def _timed(label):
#     start = time.perf_counter()
#     yield
#     end = time.perf_counter()
#     print(f"[{label}] {end - start:.4f} seconds")


def _process_deprecated(new_value, deprecated_value, new_name, deprecated_name):
    if deprecated_value is not None:
        warnings.warn(
            f"'{deprecated_name}' is deprecated and will be removed in a future version. "
            f"Please use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return deprecated_value
    return new_value


class SupercellError(Exception):
    pass


def findmax(*args):
    vals = []
    for v in args:
        if hasattr(v, "__len__"):
            vals.append(np.amax(v))
        else:
            vals.append(v)

    return np.amax(vals)


def findmin(*args):
    vals = []
    for v in args:
        if hasattr(v, "__len__"):
            vals.append(np.amin(v))
        else:
            vals.append(v)

    return np.amin(vals)


def isclose(r1, r2, rtol=1e-4):
    """Determine whether two vectors are similar

    **Parameters:**

    ``r1,r2``: 1D arrays of integer or float
        Vectors to be compared
    ``rtol``: float
        Tolerance.

    **Returns:**

    Boolean: if the euclidean distance between ``r1`` and ``r2`` is smaller than
    ``rtol``, ``True`` is returned, otherwise ``False`` is returned.
    """

    if len(r1) != len(r2):
        return False
    return np.linalg.norm(r1 - r2) < rtol


def dict_compare(d1, d2, tol=None):
    """Compare two dictionaries containing mutable objects.

    This compares two dictionaries. Two dictionaries are considered equal
    even if the position of keys differ. Handles mutable values in the dict.
    Some parts are taken from:

    https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python

    **Parameters:**

    ``d1,d2``: python dictionaries
        dictionaries to be compared

    ``tol``: float
        a small float number. If not ``None``, the comparison of dictionary
        values is regarded as a vector comparison and done with
        ``utils.isclose()``. For the meaning of ``tol``, read the documentation
        for ``rtol`` parameter of ``utils.isclose()``.

    Return: boolean
        ``True`` if dicts are equal, ``False`` if they are different.
    """
    areeq = True

    if len(d1) != len(d2):
        return False

    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    if len(d1) != len(intersect_keys):
        return False

    for k in d1_keys:
        try:
            for v1, v2 in zip(d1[k], d2[k]):
                if (isinstance(v1, list)) or (isinstance(v1, np.ndarray)):
                    for i, v in enumerate(v1):
                        if tol is None:
                            if v != v2[i]:
                                return False
                        else:
                            if not isclose(v, v2[i], tol):
                                return False
                else:
                    if tol is None:
                        if v1 != v2:
                            return False
                    else:
                        if not isclose(v1, v2, tol):
                            return False

        except TypeError:
            try:
                if _is_integrable(d1[k]):
                    if not isclose(d1[k], d2[k], tol):
                        return False
            except Exception:  # TODO: specify error type
                if type(d1[k]) is dict:
                    for subkeys in d1[k].keys():
                        if not isclose(d1[k][subkeys], d2[k][subkeys], tol):
                            return False

    return areeq


def sub_folders(root):
    """Return list of subfolders

    Parameters:

    root: string
        path of the absolute folder for which you want to get the list of
        subfolders.
    """
    # return os.walk(root).next()[1] #python 2.7
    return next(os.walk(root))[1]


def _is_integrable(s):
    """Determine whether argument is integer or can be converted to integer

    Parameters:

    s: object
        Can be any object for which we want to determine whether is integer or
        can be converted to integer.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def list_integer_named_folders(root=".", prefix="", suffix="", containing_files=None, not_containing_files=None):
    """Return integer-named folders or their original string names.

    Scans folders in `root` and detects those whose names match:

        prefix + integer + suffix

    Examples
    --------
    If the folder structure is::

        root/
            1/
            2/
            5/
            7/
            8/
            run_2/
            run_3/
            run_4/
            run_5/
            run_old/
            folderx/
            foldery/

    then with default arguments this returns::

        [1, 2, 5, 7, 8]

    If prefix="run_", this returns::

        ["run_2", "run_3", "run_4", "run_5"]

    If folders are zero-padded, e.g.::

        root/
            01/
            02/
            10/
            11/

    then this returns::

        ["01", "02", "10", "11"]

    Parameters
    ----------
    root : str
        Path of the root folder in which to scan for integer-named folders.
    prefix : str
        Required prefix before the integer part.
    suffix : str
        Required suffix after the integer part.
    containing_files : list[str]
        File names that must exist inside the returned folders.
    not_containing_files : list[str]
        File names that must not exist inside the returned folders.

    Returns
    -------
    list[int] or list[str]
        Returns integers only when:
        - prefix == ""
        - suffix == ""
        - and none of the matched folder names uses zero padding

        Otherwise returns the original folder names as strings, preserving
        zero padding.
    """
    if containing_files is None:
        containing_files = []
    if not_containing_files is None:
        not_containing_files = []

    folders = sub_folders(root)

    matches = []
    padded_found = False

    for folder in folders:
        include = True

        for fn in not_containing_files:
            if os.path.exists(os.path.join(root, folder, fn)):
                include = False
                break
        if not include:
            continue

        for fn in containing_files:
            if not os.path.exists(os.path.join(root, folder, fn)):
                include = False
                break
        if not include:
            continue

        if not folder.startswith(prefix):
            continue
        if suffix and not folder.endswith(suffix):
            continue

        start = len(prefix)
        end = len(folder) - len(suffix) if suffix else len(folder)
        d = folder[start:end]

        if _is_integrable(d):
            value = int(d)
            matches.append((value, folder))

            # detect zero padding: "01", "002", etc.
            # but do not count plain "0" as padded
            if len(d) > 1 and d[0] == "0":
                padded_found = True

    matches.sort(key=lambda x: x[0])

    # Only return integers if there is no prefix/suffix and no zero padding
    if prefix == "" and suffix == "" and not padded_found:
        return [value for value, _ in matches]

    # Otherwise preserve the original names exactly
    return [folder for _, folder in matches]


def atat_to_cell(
    file_path="lat.in",
    interpret_as="parent_lattice",
    parent_lattice=None,
    pbc=None,
    wrap=True,
):
    """Parse a ``lat.in`` or ``str.out`` file from ``ATAT``.

    ``ATAT`` users may use the input files from ``ATAT`` to perform a cluster
    expansion with **CELL**. This function allows to convert an input ``lat.in``
    file from ``ATAT`` to a ``ParentLattice`` object in CELL. One may also read
    ``str.out`` files, which are converted to ``Structure`` objects in ``CELL``.

    **Parameters:**

    ``file_path``: string
        string containing the path of the file to be parsed. The file must have
        format corresponding to a ``lat.in`` or ``str.out`` file from ATAT.
    ``interpret_as``: string or None
        Indicate how to interpret the file in ``file_path``. Possible values are:

        * ``None``:
            Three arrays are returned: ``cell``, ``r``, and ``species``.
        * ``parent_lattice``:
            The recommended value if ``file_path`` corresponds to
            a ``lat.in`` input file from ATAT. A ``ParentLattice`` **CELL**
            object will be returned.
        * ``super_cell``:
            The recommended value if ``file_path`` corresponds to
            a ``lat.in`` file with a matrix of lattice vectors (``u,v,w`` in
            `ATAT doc <https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/node35.html>`_)
            different to the identity, or if it is known that the structure is
            a periodic repetition of a parent lattice. For this option, the parent
            lattice must be provided (see ``parent_lattice`` parameter below).
            A ``SuperCell`` object will be returned.
        * ``structure``:
            The recommended value if ``file_path`` corresponds to
            a ``str.out`` file from ATAT. For this option, the parent
            lattice must be provided (see ``parent_lattice`` parameter below).
            In this case a ``Structure`` object will be returned.

    ``parent_lattice``: ParentLattice object
        If ``interpret_as`` is ``super_cell`` or ``structure``, a parent lattice
        must be provided. This must be compatible with the information in the
        ``lat.in`` file that was used to create the ``str.out`` files. See the
        examples below.
    ``pbc``: one or three bool (same as ASE's Atoms object)
        Periodic boundary conditions flags. Examples: True, False, 0, 1,
        (1, 1, 0), (True, False, False). Default value: False. The returned
        **CELL** objects will have these pbc's set up.
    ``wrap``: boolean (default:``True``)
        Wrap atomic coordinates. If pbc is ``None``, pbc is set to (1,1,1).
        Set ``wrap`` to ``False`` if structure corresponds
        to a supercell, i.e., if the second matrix of the structure definition
        in either the lat.in or str.out file is different from the identity
        matrix.

    **Returns:**

    Depending on the value of ``interpret_as``, the returned object can be python
    arrays (``interpret_as=None``), a ``ParentLattice``
    (``interpret_as="parent_lattice"``), a ``SuperCell``
    (``interpret_as="super_cell"``), or a ``Structure``
    (``interpret_as="Structure"``).

    **Examples:**

    .. todo::
        Clarify wrap option. Right now it is not guaranteed to
        work when the supercell definition in lat.in is not the
        identity matrix. This is so, because in the lat.in format
        the scaled coordinates are given in reference to the
        cartesian vectors of the parent lattice, but may define a
        supercell. Fix it such that the x matrix below referrs to
        a*b and not b as it is now.

    """
    from copy import deepcopy

    from ase import Atoms
    from ase.data import atomic_numbers

    f = open(file_path)
    lines = f.readlines()
    f.close()

    lat = []

    lat.append([])
    lnn = 0
    for ln, line in enumerate(lines):
        ls = line.split()
        if len(ls) > 0:
            if ln < 3:
                lat[0].append([float(ls[0]), float(ls[1]), float(ls[2])])
            if ln == 2:
                lat.append([])
            if ln > 2 and ln < 6:
                lat[1].append([float(ls[0]), float(ls[1]), float(ls[2])])
            if ln == 5:
                lat.append([])
            if ln >= 6:
                lat[2].append([])
                s = "".join(ls[3:]).split(",")
                lat[2][lnn].append([float(ls[0]), float(ls[1]), float(ls[2])])
                lat[2][lnn].append(s)
                lnn = lnn + 1

    struc = deepcopy(lat)

    # Write coordinate system
    parent = []
    for i in range(3):
        parent.append([struc[0][i][0], struc[0][i][1], struc[0][i][2]])

    # Write lattice
    scell = []
    for i in range(3):
        scell.append([struc[1][i][0], struc[1][i][1], struc[1][i][2]])

    # Write atoms
    x = []
    species = []
    for i in range(len(struc[2])):
        x.append([struc[2][i][0][0], struc[2][i][0][1], struc[2][i][0][2]])
        nrs = []
        for sp in struc[2][i][1]:
            if sp == "Vac" or sp == "V":
                sp = "X"
            nrs.append(atomic_numbers[sp])
        species.append(nrs)

    # np.matrix was converted to np.array, TODO: add test for this function
    x = np.array(x)
    b = np.array(parent)
    a = np.array(scell)

    ####################
    # Calculate real cartesian coordinates
    if wrap:
        from clusterx.symmetry import wrap_scaled_positions

        if pbc is None:
            pbc = (1, 1, 1)
        x = wrap_scaled_positions(x, pbc)
    r = x * b

    cell = a * b
    if interpret_as is None:
        return cell, r, species
    if interpret_as == "parent_lattice":
        from clusterx.parent_lattice import ParentLattice

        nrs = np.zeros(len(species))
        for inr, nr in enumerate(species):
            nrs[inr] = int(nr[0])

        plat = ParentLattice(Atoms(positions=r, cell=cell, numbers=nrs, pbc=pbc), sites=species, pbc=pbc)
        return plat
    if interpret_as == "super_cell":
        from clusterx.structure import Structure

        pcell = parent_lattice.get_cell()
        tmat = np.asarray(np.rint(np.dot(cell, np.linalg.inv(pcell))).astype(int))
        from clusterx.super_cell import SuperCell

        scell = SuperCell(parent_lattice, tmat)
        return scell
    if interpret_as == "structure":
        from clusterx.structure import Structure

        pcell = parent_lattice.get_cell()
        tmat = np.asarray(np.rint(np.dot(cell, np.linalg.inv(pcell))).astype(int))
        from clusterx.super_cell import SuperCell

        scell = SuperCell(parent_lattice, tmat)
        pris = scell.get_pristine()

        pos = pris.get_positions()
        new_nrs = np.zeros(len(pos))
        for ip, p in enumerate(r):
            nr = species[ip][0]
            for i2, p2 in enumerate(pos):
                if isclose(np.asarray(p).reshape(-1), p2):
                    new_nrs[i2] = nr

        struc = Structure(scell, decoration=new_nrs)
        return struc


def _is_integer_matrix(m, rnd=5):
    mi = np.around(m, rnd)
    for k in range(3):
        for el in range(3):
            # if not round(m[k,l],rnd).is_integer():
            if not mi[k, el].is_integer():
                return False
    return True


class SuperCellIndexSearcher:
    """Does the job of get_cl_idx_sc more efficiently by using a KDTree.
    This is useful for large supercells."""

    def __init__(self, positions_scell):
        from scipy.spatial import KDTree

        self.kdtree = KDTree(positions_scell)

    def get_cluster_indices(self, positions_cluster, tol=1e-3):
        """Return atom indexes of cluster points in SuperCell

        **Parameters:**

        ``cl``: npoints x 3 matrix
            matrix of cartesian coordinates of cluster points. Cluster
            positions are expected to be wrapped inside supercell ``sc``
        ``tol``: real positive number
            tolerance to determine whether cluster and atom positions are the same.
        """
        _, idxs = self.kdtree.query(positions_cluster, k=1, distance_upper_bound=tol)
        return idxs


def get_cl_idx_sc(cl, sc, method=0, tol=1e-3):
    """Return atom indexes of cluster points in SuperCell

    **Parameters:**

    ``cl``: npoints x 3 matrix
        matrix of cartesian or scaled coordinates of cluster points. Cluster
        positions are expected to be wrapped inside supercell ``sc``
    ``sc``: natoms x 3 matrix
        matrix of cartesian or scaled coordinates of supercell atomic positions.
        The type of coordinates (either cartesion or scaled) must coincide with
        that of ``cl``
    ``method``: integer
        Method to use. 0: (slow) nested for loop using numpy allclose. 1: (fast)
        calculates all distances from points in ``cl`` to atoms in ``sc``, and
        return indices for which distances are zero.
    ``tol``: real positive number
        tolerance to determine whether cluster and atom positions are the same.
    """
    from scipy.spatial.distance import cdist

    sdistances = None
    method = 1
    if method == 0:
        idxs = np.zeros(len(cl), dtype="int")
        for icl, clp in enumerate(cl):
            for isc, scp in enumerate(sc):
                if np.allclose(clp, scp, atol=tol):
                    idxs[icl] = isc
                    break

    if method == 1:
        sdistances = cdist(
            cl, sc, metric="euclidean"
        )  # Evaluate all (scaled) distances between cluster points to scell sites
        idxs = np.argwhere(np.abs(sdistances) < tol)[:, 1]  # Atom indexes of the transformed cluster

    return idxs


def add_noise(v, noise_level):
    """Add randomly distributed noise to vector coordinates

    To each coordinate of the input vector ``v``, random noise uniformly
    distributed between -``noise_level`` and ``noise_level`` is added. The input
    vector ``v`` is left unchanged. The modified vector is returned.

    **Parameters:**

    ``v``: list of floats
        The input vector

    ``noise_level``: float
        Width of the uniform distribution used to add noise.
    """
    import random

    energies = []
    for e in v:
        energies.append(e + random.uniform(-1, 1) * noise_level)
    return energies


def calculate_trafo_matrix(pcell, scell, rnd=5):
    """Calculate integer transformation matrix given a primitive cell and a super-cell

    If :math:`S` and :math:`V` are, respectively, a matrix whose rows are the cartesian coordinates
    of the parent lattice vectors and a matrix whose rows are the cartesian coordinates
    of the super-cell lattice vectors; then, this function returns the matrix :math:`P=SV^{-1}`.
    If the resulting matrix is not integer (see ``rnd`` parameter), then ``None`` is returned.

    **Parameters:**

    ``pcell``: 3x3 array of float
        The rows of this matrix correspond to the cartesian coordinates of a parent lattice

    ``scell``: 3x3 array of float
        The rows of this matrix correspond to the cartesian coordinates of a supercell

    ``rnd``: integer (optional)
         The matrix :math:`P=SV^{-1}` is rounded to ``rnd`` decimal places and checked for integrity.
    """
    tmat = np.dot(scell, np.linalg.inv(pcell))
    if _is_integer_matrix(tmat, rnd):
        return np.asarray(np.rint(tmat).astype(int))
    else:
        return None


def _str_grep(input_str, search_str, prepend=""):
    out_str = ""
    for el in input_str.split("\n"):
        if search_str in el:
            out_str = out_str + prepend + el.strip() + "\n"

    return out_str.rstrip()


def mgrep(fpath, search_array, prepend="", root="."):
    """
    Grep strings in file and return matching lines.

    **Parameters:**

    ``fpath``: string
        File path to grep
    ``search_array``: array of strings
        Each element of the array is an string to grep in ``fpath``.
    ``prepend``: string
        prepend string ``prepend`` to each matching line.
    ``root``: string
        File to grep should be in ``root/fpath``.
    """
    abs_path = os.path.join(root, fpath)
    out_str = ""
    if os.path.isfile(abs_path):
        fstr = open(abs_path).read()

        for attr in search_array:
            ostr = _str_grep(fstr, attr, prepend=prepend)

            if ostr != "":
                out_str = out_str + prepend + ostr.strip() + "\n"

    return out_str.rstrip()


def parent_lattice_to_atat(plat, out_fname="lat.in", for_str=False):
    """Serializes ParentLattice object to ATAT input file

    **Parameters:**

    ``plat``: ParentLattice object
        ParentLattice object to be serialized
    ``out_fname``: string
        Output file path
    """
    cell = plat.get_cell()
    positions = plat.get_scaled_positions()
    if for_str:
        sites = plat.get_atomic_numbers()
    else:
        sites = plat.get_sites()

    f = open(out_fname, "w+")

    for cellv in cell:
        f.write("%2.12f\t%2.12f\t%2.12f\n" % (cellv[0], cellv[1], cellv[2]))

    f.write("1.000000000000\t0.000000000000\t0.000000000000\n")
    f.write("0.000000000000\t1.000000000000\t0.000000000000\n")
    f.write("0.000000000000\t0.000000000000\t1.000000000000\n")

    for i, pos in enumerate(positions):
        stri = "%2.12f\t%2.12f\t%2.12f\t" % (pos[0], pos[1], pos[2])
        if for_str:
            stri = stri + "%s\n" % cs[sites[i]]
        else:
            if len(sites[i]) > 1:
                for z in sites[i][:-1]:
                    stri = stri + "%s," % cs[z]
            stri = stri + "%s\n" % cs[sites[i][-1]]

        f.write(stri)

    f.close()


class Exponential:
    """Basic exponential object of type ``coefficient`` * ``x`` ^ ``exponent``.
    Numerically evaluated using the method ``evaluate`` ( ``x`` ) .

    **Parameters:**

    ``exponent``: exponent of the exponential

    ``coefficient``: coefficient of the exponential

    """

    def __init__(self, exponent, coefficient=1):
        self.exponent = exponent
        self.coefficient = coefficient

    def evaluate(self, x):
        return self.coefficient * np.power(x, self.exponent)

    def divide_scalar(self, value):
        self.coefficient = self.coefficient / value

    def multiply_scalar(self, scalar):
        self.coefficient = self.coefficient * scalar


class PolynomialFunction:
    """Polynomial function, build from ``Exponential()`` ."""

    def __init__(self):
        self.exponentials = []

    def add_exponential(self, order, coefficient=1):
        in_sum = False
        for exponential in self.exponentials:
            if exponential.exponent == order:
                exponential.coefficient += coefficient
                in_sum = True
                break
        if not in_sum:
            self.exponentials.append(Exponential(order, coefficient))

    def clear_exponentials(self):
        for exponential in self.exponentials:
            if abs(exponential.coefficient) < 10 ** (-15):
                self.exponentials.remove(exponential)

    def evaluate(self, x):
        value = 0
        for exponential in self.exponentials:
            value += exponential.evaluate(x)
        return value

    def normalize(self, scalar_product, m):
        length = scalar_product(self.evaluate, self.evaluate, m)
        length = np.sqrt(length)
        for exponential in self.exponentials:
            exponential.divide_scalar(length)

    def multiply_scalar(self, scalar):
        for exponential in self.exponentials:
            exponential.multiply_scalar(scalar)

    def add_polynomial_function(self, polynomial_function):
        for exponential in polynomial_function.exponentials:
            self.add_exponential(exponential.exponent, exponential.coefficient)

    def print_polynomial(self):
        outstring = ""
        for exponential in self.exponentials:
            outstring += str(exponential.coefficient) + " *  x^" + str(exponential.exponent) + " + "
        outstring = outstring[:-3]
        print(outstring)


class PolynomialBasis:
    """Polynomial basis, constructed from several ``PolynomialFunction()``.
    Constructs orthonormal basis sets using ``scalcar_product`` .
    When initialized, all orthonormal basis sets to the order ``max_order`` are generated.

    **Parameters**

    ``max_order``: Maximal order to which the basis set is initialized.

    ``symmetric``: Defines if the scalar product uses sigmas symmetrized around 0 or ascending from 0.

    """

    def __init__(self, max_order=10, symmetric=False):
        self.m = max_order
        self.symmetric = symmetric
        self.basis_function_set = {}
        for order in range(1, max_order + 1):
            self.basis_function_set[order] = self.construct(order)

    def construct(self, m):
        basis_functions = []
        for degree in range(m):
            new_function = PolynomialFunction()
            new_function.add_exponential(degree)
            chi = Exponential(degree)
            for basis_function in basis_functions:
                overlap = self.scalar_product(basis_function.evaluate, chi.evaluate, m)
                overlap = overlap * (-1)
                summand = copy.deepcopy(basis_function)
                summand.multiply_scalar(overlap)
                new_function.add_polynomial_function(summand)
            new_function.normalize(self.scalar_product, m)
            new_function.clear_exponentials()
            basis_functions.append(new_function)
        return basis_functions

    def scalar_product(self, function1, function2, m):
        """
        returns the result of the scalar product 1/m sum_{sigma} f_1(sigma) * f_2(sigma)
        """
        scaling = 1 / m
        if self.symmetric:
            sigmas = [x for x in range(-int(m / 2), int(m / 2) + 1)]
            if m % 2 == 0:
                sigmas.remove(0)
        else:
            sigmas = [x for x in range(m)]
        scalar_product = 0
        for sigma in sigmas:
            scalar_product += function1(sigma) * function2(sigma)
        scalar_product = scalar_product * scaling
        return scalar_product

    def print_basis_functions(self, m):
        for function in self.basis_function_set[m]:
            function.print_polynomial()

    def evaluate(self, alpha, sigma, m):
        if m > self.m:
            self.basis_function_set[m] = self.construct(m)
        return self.basis_function_set[m][int(alpha)].evaluate(sigma)


def poppush(x, val):
    """Left-shift array one position and writes new value to the right. Returns average.

    **Parameters:**
    ``x``: numpy array
    ``val``: int or float
    """
    x[:-1] = x[1:]
    x[-1] = val
    return x.mean()


def sort_atoms(atoms, key=(2, 1, 0)):
    """Return atoms object with sorted atomic coordinates

    The default sorting is: increasing z-coordinate first, increasing
    y-coordinate second, increasing x-coordinate third. Useful to get
    well ordered slab structures, for instance. Sorting can be changed
    by appropriately setting the ``key`` argument, with the same
    effect as in::

        from operator import itemgetter
        sp = sorted(p, key=itemgetter(2,1,0))

    where p is a Nx3 array of vector coordinates.
    """
    if key is None:
        return atoms
    else:
        nrs = atoms.get_atomic_numbers()
        poss = atoms.get_positions()
        pn = []
        for p, n in zip(poss, nrs):
            pn.append([p[0], p[1], p[2], n])
        from operator import itemgetter

        import numpy as np

        _pn = sorted(pn, key=itemgetter(*key))
        _poss = np.delete(np.array(_pn), 3, 1)
        _nrs = np.delete(np.array(_pn), [0, 1, 2], 1).flatten()
        from ase import Atoms

        return Atoms(positions=_poss, numbers=_nrs, cell=atoms.get_cell(), pbc=atoms.get_pbc())


def remove_vacancies(at):
    """Remove every Atom containing 'X' as species symbol or 0 as atomic number
    from an Atoms object and return the resulting Atoms object.

    **Parameters:**
    ``at``: Atoms object

    .. todo::
        find cleaner way to do this...
    """
    from ase import Atoms

    positions = []
    numbers = []
    indices = []
    tags = []
    tags0 = at.get_tags()
    momenta = []
    momenta0 = at.get_momenta()
    masses = []
    masses0 = at.get_masses()

    for i, atom in enumerate(at):
        nr = atom.get("number")
        if nr != 0:
            indices.append(i)
            positions.append(atom.get("position"))
            tags.append(tags0[i])
            momenta.append(momenta0[i])
            masses.append(masses0[i])
            numbers.append(nr)

    return Atoms(
        cell=at.get_cell(),
        pbc=at.get_pbc(),
        numbers=numbers,
        positions=positions,
        calculator=at.calc,
        tags=tags,
        momenta=momenta,
        masses=masses,
        celldisp=at.get_celldisp(),
        constraint=at.constraints,
        info=at.info,
    )


def make_supercell(prim, P, wrap=True, tol=1e-5):
    r"""Generate a supercell by applying a general transformation (*P*) to
    the input configuration (*prim*).

    This function is a modified version of ASEs build/supercells.py.
    The modification here fixes a bug in ASEs implementation, introduced in
    ASEs version 3.18.0: for certain transformation matrices, the determinant of
    the matrix is negative, and the function exits with error, since a negative
    number of atoms is obtained.
    An example script demonstrating the error is::

       from ase.build import make_supercell
       from ase.atoms  import Atoms

       prim = Atoms(symbols='Cu',
             pbc=True,
             cell=[[0.0, 1.805, 1.805], [1.805, 0.0, 1.805], [1.805, 1.805, 0.0]])

       sc = make_supercell(prim, P = [[ 2,  2, -2],[ 2, -2,  2],[-2,  2,  2]], wrap = True, tol = 1e-05)

    The release 3.22.0 of ASE still contains the bug.
    It was informed to ASEs developers on the 3.7.2021

    The transformation is described by a 3x3 integer matrix
    `\mathbf{P}`. Specifically, the new cell metric
    `\mathbf{h}` is given in terms of the metric of the input
    configuration `\mathbf{h}_p` by `\mathbf{P h}_p =
    \mathbf{h}`.

    **Parameters:**

    ``prim``: ASE Atoms object
        Input configuration.
    ``P``: 3x3 integer matrix
        Transformation matrix `\mathbf{P}`.
    ``wrap``: bool
        wrap in the end
    ``tol``: float
        tolerance for wrapping
    """

    supercell_matrix = P
    supercell = clean_matrix(supercell_matrix @ prim.cell)

    # cartesian lattice points
    lattice_points_frac = lattice_points_in_supercell(supercell_matrix)
    lattice_points = np.dot(lattice_points_frac, supercell)

    superatoms = Atoms(cell=supercell, pbc=prim.pbc)

    for lp in lattice_points:
        shifted_atoms = prim.copy()
        shifted_atoms.positions += lp
        superatoms.extend(shifted_atoms)

    # check number of atoms is correct
    n_target = np.abs(int(np.round(np.linalg.det(supercell_matrix) * len(prim))))
    if n_target != len(superatoms):
        msg = "Number of atoms in supercell: {}, expected: {}".format(n_target, len(superatoms))
        raise SupercellError(msg)

    if wrap:
        superatoms.wrap(eps=tol)

    return superatoms


def decorate_supercell(
    scell,
    atoms: Atoms,
    tol: Optional[float] = 1e-5,
    tols: Optional[List[float]] = None,
    wrap: bool = False,
    wrap_eps: float = 0.05,
):
    """
    Create a clusterx.Structure by decorating a SuperCell with species taken
    from an ASE Atoms object.

    For each site in `scell`, this function finds a matching atom in `atoms`
    whose Cartesian position is within a distance tolerance (Euclidean
    distance).
    The corresponding atomic number is then used as the decoration. If the
    supercell has a substitutional sublattice that allows vacancies (species
    number 0, chemical symbol ``X``), missing atoms on that sublattice are
    decorated as vacancies.

    Parameters
    ----------
    scell : SuperCell
        The supercell object (from clusterx) providing the lattice and sites.
    atoms : Atoms
        ASE Atoms instance containing the atomic positions and species.
    tol : float, optional
        Distance tolerance for matching positions. Used only when ``tols``
        is ``None``. Default is 1e-5.
    tols : list[float], optional
        If provided, ``decorate_supercell`` ignores ``tol`` and tries each
        tolerance in increasing order until a valid decoration is found.
    wrap : bool, optional
        Whether to wrap ASE atomic positions into the unit cell before matching.
        Passed directly to `atoms.get_positions(wrap=wrap)`.
    wrap_eps : float, optional
        ``eps`` argument forwarded to position wrapping.

    Returns
    -------
    Structure
        A clusterx.Structure object decorated with atomic numbers from `atoms`.

    Raises
    ------
    ValueError
        If no valid decoration is found for the tested tolerance(s).
    """
    from clusterx.structure import Structure

    # Pre-fetch positions and atomic numbers for efficiency
    sc_positions = np.asarray(scell.get_positions())
    sc_tags = np.asarray(scell.get_tags(), dtype=int)
    at_positions = np.asarray(atoms.get_positions(wrap=wrap, eps=wrap_eps))
    at_numbers = np.asarray(atoms.get_atomic_numbers())
    expected_vacancies = len(sc_positions) - len(at_positions)

    if expected_vacancies < 0:
        raise ValueError(
            f"Atoms object contains more atoms ({len(at_positions)}) than "
            f"supercell sites ({len(sc_positions)})."
        )

    sublattice_types = scell.get_sublattice_types()
    substitutional_tags = {int(tag) for tag in scell.get_substitutional_tags()}
    vacancy_tags = {
        int(tag)
        for tag, species in sublattice_types.items()
        if int(tag) in substitutional_tags and 0 in np.asarray(species, dtype=int)
    }

    if expected_vacancies > 0 and not vacancy_tags:
        raise ValueError(
            f"Atoms object is missing {expected_vacancies} atom(s), but the "
            "supercell has no substitutional sublattice allowing vacancies "
            "(species number 0 / symbol X)."
        )

    n_vacancy_sites = sum(int(tag) in vacancy_tags for tag in sc_tags)
    if expected_vacancies > n_vacancy_sites:
        raise ValueError(
            f"Atoms object is missing {expected_vacancies} atom(s), but the "
            f"supercell has only {n_vacancy_sites} vacancy-capable site(s)."
        )

    def _format_no_match_error(
        tol_value: float,
        i_p_sc: int,
        p_sc: np.ndarray,
        dists: np.ndarray,
        no_unused_match: bool = False,
    ) -> str:
        if len(dists) == 0:
            return (
                f"No matching atom found within tolerance {tol_value} "
                f"for:\n"
                f"supercell position {i_p_sc}: {p_sc}.\n"
                "The atoms object is empty."
            )

        mind = min(dists)
        i_mind = np.where(dists == min(dists))[0][0]
        match_text = (
            "No unused matching atom" if no_unused_match else "No matching atom"
        )
        return (
            f"{match_text} found within tolerance {tol_value} "
            f"for:\n"
            f"supercell position {i_p_sc}: {p_sc}.\n"
            f"atoms position {i_mind}:     {at_positions[i_mind]}\n"
            f" Smallest distance "
            f"to this position is {mind}."
        )

    def _decorate_with_tol(tol_value: float) -> Structure:
        decorations = []
        used_atoms = set()
        vacancy_count = 0

        for i_p_sc, p_sc in enumerate(sc_positions):
            # Compute distances from this supercell site to all atoms
            deltas = at_positions - p_sc
            dists = np.linalg.norm(deltas, axis=1)

            # Find atoms within tolerance
            candidates = np.where(dists < tol_value)[0]
            if len(candidates) > 1:
                candidates = candidates[np.argsort(dists[candidates])]

            i_match = None
            for candidate in candidates:
                candidate = int(candidate)
                if candidate not in used_atoms:
                    i_match = candidate
                    break

            if i_match is None:
                if int(sc_tags[i_p_sc]) in vacancy_tags:
                    decorations.append(0)
                    vacancy_count += 1
                    continue

                raise ValueError(
                    _format_no_match_error(
                        tol_value,
                        i_p_sc,
                        p_sc,
                        dists,
                        no_unused_match=len(candidates) > 0,
                    )
                )

            decorations.append(int(at_numbers[i_match]))
            used_atoms.add(i_match)

        if len(decorations) != len(scell):
            raise ValueError(
                f"Number of decorations ({len(decorations)}) does not match "
                f"number of sites in supercell ({len(scell)}). You may need "
                f"to increase tol (currently {tol_value})."
            )

        if vacancy_count != expected_vacancies:
            raise ValueError(
                f"Detected {vacancy_count} vacancy site(s) at tolerance {tol_value}, "
                f"but expected {expected_vacancies} from atom counts "
                f"({len(sc_positions)} supercell sites, {len(at_positions)} atoms)."
            )

        return Structure(scell, decorations)

    if tols is not None:
        if len(tols) == 0:
            raise ValueError("Parameter 'tols' was provided but is empty. Provide at least one tolerance value.")

        sorted_tols = sorted(tols)
        last_error = None

        for tol_value in sorted_tols:
            try:
                return _decorate_with_tol(tol_value)
            except ValueError as err:
                last_error = err

        raise ValueError(
            f"Could not decorate supercell with any tolerance in {sorted_tols}. Last error: {last_error}"
        ) from last_error

    if tol is None:
        raise ValueError("Parameter 'tol' cannot be None when 'tols' is not provided.")

    return _decorate_with_tol(tol)


def super_structure(struc0, d):
    """Create a super structure

    This function takes a ``Structure`` instance (``struc0``) and creates a new structure
    which is obtained as the periodic repetition of the original structure
    along its unit cell vectors. The number of repetitions along each cell vector is given
    by the three components of the input integer vector ``d``.

    **Parameters:**

    ``struc0``: Structure object
        Original structure for the superstructure.
    ``d``: int, three-component integer array, or 3x3 integer array
        The super structure is obtained by the transformation d S, with d a
        3x3 matrix of integer and S the supercell cell vectors.
    """
    from ase.build import make_supercell

    from clusterx.super_cell import SuperCell

    if np.shape(d) == ():
        n = np.zeros((3, 3), int)
        np.fill_diagonal(n, [d, d, d])
    elif np.shape(d) == (3,):
        n = np.zeros((3, 3), int)
        np.fill_diagonal(n, d)
    elif np.shape(d) == (3, 3):
        n = np.array(d)
    else:
        print("ERROR (clusterx.utils.super_structure()): ")

    p0 = struc0.get_supercell().get_transformation()
    p1 = n @ p0

    atoms0 = struc0.get_atoms()
    atoms1 = make_supercell(atoms0, n)

    scell1 = SuperCell(struc0.get_parent_lattice(), p1)

    return decorate_supercell(scell1, atoms1)


def sset_equivalence_check(
    sset,
    to_primitive=True,
    cpool=None,
    basis="trigonometric",
    comat=None,
    pretty_print=False,
):
    """Find equivalent structures in a StructuresSet object

    Equivalence is determined *i)* in terms of symmetry between structures
    or *ii)* in terms of cluster basis representation (if a ClustersPool object
    or a correlation matrix is given).

    In the first case, the `SymmetryEquivalenceCheck` tool of ASE
    <https://wiki.fysik.dtu.dk/ase/ase/utils.html#ase.utils.structure_comparator.SymmetryEquivalenceCheck>`_ is used.

    **Parameters:**

    ``sset``: StructuresSet object
        The structures set object to be analyzed.

    ``to_primitive``: Boolean (default: ``True``)
        If ``True`` the structures are reduced to their primitive cells.
        This feature requires ``spglib`` to installed
        (*cf.* `ASE's SymmetryEquivalenceCheck <https://wiki.fysik.dtu.dk/ase/ase/utils.html?highlight=to_primitive#ase.utils.structure_comparator.SymmetryEquivalenceCheck>`_)

    ``cpool``: ClustersPool object (default: ``None``)
        This parameter is optional. If given, the equivalence of a pair of structures
        is determined according to their cluster basis representation: two
        structures with the same cluster correlations for the clusters in ``cpool``
        are considered equivalent.

    ``basis``: string (default: ``"trigonometric"``)
        Only used if ``cpool`` is not ``None``. Site basis functions used in the determination
        of cluster correlations.

    ``comat``: 2D array of floats (default: ``None``)
        It overrides ``cpool`` and ``basis``. If a correlation matrix for the
        structures set sset was pre-computed, the equivalence in terms of cluster
        basis representation is performed by comparing the rows of this matrix (each
        row must correspond to a structure in the structures set).


    **Returns:**
    Returns a dictionary. The keys (k) are structure indices of unique representative structures,
    and the values (v) are arrays of integer, indicating all structure indices equivalent to k
    (containing k itself too). For instance, the dictionary::

        {"0": [0, 1, 3, 8, 9],
         "2": [2, 5, 6],
         "4": [4, 7]}

    | indicates that in the structures set with indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] there are
      just three distinct structures. These can be represented by strucutres 0, 2 and 4.
      The structures [0, 1, 3, 8, 9] are all equivalent, etc.
    | Notice that here equivalence is used in the sense explained above: It is symmetrical
      equivalence only if ``cpool`` and ``comat`` are None.
    """
    import numpy as np

    comp = None
    from clusterx.utils import isclose

    if cpool is None and comat is None:
        from ase.utils.structure_comparator import SymmetryEquivalenceCheck

        comp = SymmetryEquivalenceCheck(to_primitive=to_primitive)
    elif comat is None:
        from clusterx.correlations import CorrelationsCalculator

        ccalc = CorrelationsCalculator(basis=basis, parent_lattice=sset.get_parent_lattice(), clusters_pool=cpool)
        comat = ccalc.get_correlation_matrix(sset)

    nstr = len(sset)

    crossedout = []
    id_str_list = {}
    for i in range(nstr):
        if i not in crossedout:
            crossedout.append(i)
            subset = [i]

            if comat is not None:
                corr_i = comat[i, :]
            else:
                atoms_i = sset[i].get_atoms()

            for j in range(i + 1, nstr):
                if comat is not None:
                    corr_j = comat[j, :]
                    check = isclose(corr_i, corr_j)
                else:
                    atoms_j = sset[j].get_atoms()
                    check = comp.compare(atoms_i, atoms_j)

                if check:
                    crossedout.append(j)
                    subset.append(j)

            id_str_list[i] = np.array(subset, dtype="i4")

    if pretty_print:
        print(id_str_list)

    return id_str_list


def ssets_equivalence_check(
    sset1,
    sset2=None,
    to_primitive=True,
    pretty_print=False,
):
    """Find equivalent structures within or between StructuresSet objects.

    If only ``sset1`` is provided, structures are compared with other structures
    in the same set and the return value follows ``sset_equivalence_check``:
    each key is a representative index and each value contains all equivalent
    indices from ``sset1``.

    If ``sset2`` is provided, each structure in ``sset1`` is compared with all
    structures in ``sset2``. In that case every key is an index from ``sset1``
    and every value contains the matching indices from ``sset2``.

    Structures are compared only when their
    ``Atoms.get_chemical_formula(mode="hill", empirical=True)`` strings are
    identical.
    """
    import numpy as np
    from ase.utils.structure_comparator import SymmetryEquivalenceCheck

    comp = SymmetryEquivalenceCheck(to_primitive=to_primitive)
    compare_same_set = sset2 is None
    if sset2 is None:
        sset2 = sset1

    atoms1 = [structure.get_atoms() for structure in sset1]
    atoms2 = atoms1 if compare_same_set else [structure.get_atoms() for structure in sset2]
    formulas1 = [atoms.get_chemical_formula(mode="hill", empirical=True) for atoms in atoms1]
    formulas2 = (
        formulas1 if compare_same_set else [atoms.get_chemical_formula(mode="hill", empirical=True) for atoms in atoms2]
    )

    id_str_list = {}

    if compare_same_set:
        crossedout = []
        for i, atoms_i in enumerate(atoms1):
            if i in crossedout:
                continue

            crossedout.append(i)
            subset = [i]
            formula_i = formulas1[i]

            for j in range(i + 1, len(atoms1)):
                if formulas1[j] != formula_i:
                    continue

                if comp.compare(atoms_i, atoms1[j]):
                    crossedout.append(j)
                    subset.append(j)

            id_str_list[i] = np.array(subset, dtype="i4")
    else:
        for i, atoms_i in enumerate(atoms1):
            subset = []
            formula_i = formulas1[i]

            for j, atoms_j in enumerate(atoms2):
                if formulas2[j] != formula_i:
                    continue

                if comp.compare(atoms_i, atoms_j):
                    subset.append(j)

            id_str_list[i] = np.array(subset, dtype="i4")

    if pretty_print:
        print(id_str_list)

    return id_str_list


def atoms_equivalence_check(atoms, to_primitive=True, pretty_print=False):
    """Find equivalent structures in an array of Atoms objects

    Equivalence is determined in terms of symmetry between structures

    The `SymmetryEquivalenceCheck` tool of ASE
    <https://wiki.fysik.dtu.dk/ase/ase/utils.html#ase.utils.structure_comparator.SymmetryEquivalenceCheck>`_ is used.

    **Parameters:**

    ``atoms``: array of Atoms objects
        The structures to be analyzed.

    ``to_primitive``: Boolean (default: ``True``)
        If ``True`` the structures are reduced to their primitive cells.
        This feature requires ``spglib`` to installed
        (*cf.* `ASE's SymmetryEquivalenceCheck <https://wiki.fysik.dtu.dk/ase/ase/utils.html?highlight=to_primitive#ase.utils.structure_comparator.SymmetryEquivalenceCheck>`_)


    **Returns:**
    Returns a dictionary. The keys (k) are structure indices of unique representative structures,
    and the values (v) are arrays of integer, indicating all structure indices equivalent to k
    (containing k itself too). For instance, the dictionary::

        {"0": [0, 1, 3, 8, 9],
         "2": [2, 5, 6],
         "4": [4, 7]}

    | indicates that in the structures set with indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] there are
      just three distinct structures. These can be represented by strucutres 0, 2 and 4.
      The structures [0, 1, 3, 8, 9] are all equivalent, etc.
    | Notice that here equivalence is used in the sense explained above: It is symmetrical
      equivalence only if ``cpool`` and ``comat`` are None.
    """
    import numpy as np
    from ase.utils.structure_comparator import SymmetryEquivalenceCheck

    comp = SymmetryEquivalenceCheck(to_primitive=to_primitive)

    nstr = len(atoms)

    crossedout = []
    id_str_list = {}
    for i in range(nstr):
        if i not in crossedout:
            crossedout.append(i)
            subset = [i]

            atoms_i = atoms[i]

            for j in range(i + 1, nstr):
                atoms_j = atoms[j]
                check = comp.compare(atoms_i, atoms_j)

                if check:
                    crossedout.append(j)
                    subset.append(j)

            id_str_list[i] = np.array(subset, dtype="i4")

    if pretty_print:
        print(id_str_list)

    return id_str_list


def report_sset_equivalence_check(sset, sset_equivalence_check_output, property_name=None, tol=0.0):
    """Generate report of equivalent structures

    Writes to files: ``sset_unique_sym.json`` and ``sset_unique_gss.json``.

    The first contains the structures whose index are given by the keys of the dictionary
    returned by ``sset_equivalence_check()``.

    The second contains the structures of every equivalence subset where the value of
    ``property_name`` is minimal. So, if the property is an energy, the final set will contain all
    lowest energy structures of every equivalence subset.

    """
    import operator

    folders = sset.get_folders()

    id_str_list = sset_equivalence_check_output

    unique = []
    for k, v in id_str_list.items():
        unique.append(v[0])

    sset_unique = sset.get_subset(unique)
    sset_unique.serialize("sset_unique_sym.json", overwrite=True)

    unique_gss = []  # Collect the lowest energy structure from each subset
    pvals = sset.get_property_values(property_name)
    for k, v in id_str_list.items():
        if len(v) > 1:
            pvals_subset = operator.itemgetter(*v)(pvals)
            i_min = np.argmin(pvals_subset)
            # unique_gss.append(i_min)
            unique_gss.append(v[i_min])
        else:
            unique_gss.append(v[0])

    sset_unique_gss = sset.get_subset(unique_gss)
    sset_unique_gss.serialize("sset_unique_gss.json", overwrite=True)

    decim = 5
    tol = tol
    print("Structure indices start from 1 (corresponding to sset[0]).")
    print(f"floats shown below are rounded to {decim} decimals.")
    print(f"Show only sets where maximum energy variation is larger than {tol}.")
    for k, v in id_str_list.items():
        subset = sset.get_subset(v)
        pvals = subset.get_property_values(property_name)
        ediff = round(np.amax(pvals) - np.amin(pvals), decim)
        if len(v) > 1 and ediff >= tol:
            print("\n========================================================")
            # subset = sset.get_subset(v)
            subset.serialize(f"repetitions-{k}.json")
            # pvals = subset.get_property_values('energy_mixing_atom')
            # print(k, pvals)
            # print(k, round(np.amax(pvals)-np.amin(pvals),5), np.around(pvals, decimals = 5))
            print(f"Found {len(pvals)} structures possibly equivalent to structure:")
            print(k + 1)
            print("List of str indices:")
            print(np.array(id_str_list[k]) + 1)
            print("Energies for these structures:")
            print(np.around(pvals, decimals=decim))
            print("Difference between maximum and minimum energies:")
            # print(round(np.amax(pvals)-np.amin(pvals),decim))
            print(ediff)
            print("Folders:")
            print(operator.itemgetter(*id_str_list[k])(folders))
            # print(folders)
            # print(map(folders.__getitem__, np.array(id_str_list[k]).tolist()))


def normalize_shape_input(p):
    """
    Normalize the supercell shape input into a 3x3 matrix.

    Parameters:
    -----------
    p: None, int, list[int], list[list[int]], or np.ndarray
        The supercell shape specification.

    Returns:
    --------
    np.ndarray or None
        A 3x3 numpy array if sc_shape is specified, otherwise None.
    """
    if p is None:
        return None
    p = np.array(p)
    if p.ndim == 0:
        return np.diag([p, p, p])
    elif p.shape == (1,):
        return np.diag([p[0], 1, 1])
    elif p.shape == (2,):
        return np.diag([p[0], p[1], 1])
    elif p.shape == (3,):
        return np.diag(p)
    elif p.shape == (2, 2):
        sc_shape = np.eye(3, dtype=int)
        sc_shape[0:2, 0:2] = p
        return sc_shape
    elif p.shape == (3, 3):
        return p
    else:
        raise ValueError("Invalid supercell shape, must be (), (3,), or (3,3).")


def normalize_nsubs_list(
    nsubs_list: Optional[Union[int, List[int], List[List[int]]]],
) -> Optional[List[List[int]]]:
    """
    Normalize the input `nsubs_list` into a list of lists of integers.

    Parameters:
        nsubs_list (Optional[Union[int, List[int], List[List[int]]]]):
            The input to normalize. It can be:
            - An integer (e.g., 3)
            - A list of integers (e.g., [3, 4, 7])
            - A list of lists of integers (e.g., [[3, 4, 7], [1, 2]])
            - None

    Returns:
        Optional[List[List[int]]]:
            The normalized list of lists of integers, or None if input is None.

    Examples:
        >>> normalize_nsubs_list(3)
        [[3]]

        >>> normalize_nsubs_list([3, 4, 7])
        [[3, 4, 7]]

        >>> normalize_nsubs_list([[3, 4, 7], [1, 2]])
        [[3, 4, 7], [1, 2]]

        >>> normalize_nsubs_list(None)
        None
    """
    if nsubs_list is None:
        return None
    if isinstance(nsubs_list, int):
        return [[nsubs_list]]
    if isinstance(nsubs_list, list):
        if all(isinstance(i, int) for i in nsubs_list):
            return [nsubs_list]
        if all(isinstance(i, list) and all(isinstance(j, int) for j in i) for i in nsubs_list):
            return nsubs_list
    raise ValueError("Invalid format for nsubs_list")


def rescale_by_first_nonzero(A):
    """
    Find the first nonzero element a_ij0 (row-major order) such that
    closest_int_to(a_ij0) != 0, then compute

        factor = a_ij0 / closest_int_to(a_ij0)

    and return A scaled by 1/factor.

    If no such element exists, raise an error.
    """

    A = np.asarray(A)
    flat = A.ravel()

    for val in flat:
        if val == 0:
            continue

        closest_int = int(np.round(val))

        # Skip if rounding gives zero
        if closest_int == 0:
            continue

        factor = val / closest_int
        return A / factor, factor

    raise ValueError("No suitable element found (all nonzero elements round to zero).")


def check_and_convert_integer_matrix(A, tol=1e-8):
    """
    Check whether all elements of a matrix are (numerically) integers,
    and return (is_integer_matrix, integer_matrix).

    Elements like 1.0 are considered integers.
    A tolerance is used to handle floating point noise.

    Parameters
    ----------
    A : np.ndarray
        Input array (e.g., 3x3)
    tol : float
        Tolerance for checking integer closeness

    Returns
    -------
    (bool, np.ndarray)
        - True/False whether all elements are integers
        - Matrix converted to integers (rounded)
    """

    A = np.asarray(A)

    # Round to nearest integer
    A_rounded = np.round(A)

    # Check if all elements are close to their rounded values
    is_integer = np.all(np.abs(A - A_rounded) < tol)

    return is_integer, A_rounded.astype(int)


def find_sc_shape(
    scell,
    pcell,
    threshold_length=1e-5,
    tol_int=1e-3,
    true_rescale=False,
    n_atoms_scell=None,
    n_atoms_pcell=None,
):
    """Determine the integer transformation matrix (shape) relating a supercell to a primitive cell.

    Given lattice matrices ``scell`` and ``pcell``, the *shape* is defined as

    .. math::

        S = P_{\\text{sc}} \\cdot P_{\\text{pc}}^{-1}

    where rows of each matrix are lattice vectors.
    The raw shape ``S`` is then rescaled so that its first significant
    element is closest to an integer, and finally checked for integer-valuedness.

    Parameters
    ----------
    scell : np.ndarray, shape (3, 3)
        Supercell lattice matrix.
    pcell : np.ndarray, shape (3, 3)
        Primitive cell lattice matrix.
    threshold_length : float, optional
        Matrix elements whose absolute value is below this threshold are
        zeroed before the inversion.  Default is ``1e-8``.
    tol_int : float, optional
        Tolerance passed to ``check_and_convert_integer_matrix`` when
        deciding whether the rescaled shape elements are integers.
        Default is ``1e-3``.
    n_atoms_scell : int, optional
        Number of atoms in the supercell.  When both ``n_atoms_scell`` and
        ``n_atoms_pcell`` are provided, the function checks that
        ``n_atoms_scell / n_atoms_pcell`` agrees with
        ``abs(det(shape_int))``.  A ``ValueError`` is raised if they
        disagree.
    n_atoms_pcell : int, optional
        Number of atoms in the primitive cell.  See ``n_atoms_scell``.

    Returns
    -------
    is_shape_int : bool
        ``True`` if the rescaled shape matrix is integer-valued within
        ``tol_int``.
    shape_int : np.ndarray of int, shape (3, 3)
        The rescaled shape matrix rounded to the nearest integers.
    shape_rescaled : np.ndarray, shape (3, 3)
        ``shape`` divided by the scale factor determined by
        ``rescale_by_first_nonzero``.
    shape : np.ndarray, shape (3, 3)
        Raw floating-point result of ``scell @ inv(pcell)`` (after
        threshold zeroing).
    """
    pcell[np.abs(pcell) < threshold_length] = 0.0
    scell[np.abs(scell) < threshold_length] = 0.0

    shape = scell @ np.linalg.inv(pcell)

    shape_rescaled, factor = rescale_by_first_nonzero(shape) if true_rescale else (shape.copy(), 1.0)
    is_shape_int, shape_int = check_and_convert_integer_matrix(shape_rescaled, tol=tol_int)

    sc_index = abs(int(np.round(np.linalg.det(shape_int))))

    if n_atoms_scell is not None and n_atoms_pcell is not None:
        ratio = n_atoms_scell // n_atoms_pcell
        if ratio != sc_index:
            shape_str = np.array2string(
                shape,
                precision=10,
                suppress_small=False,
                separator=", ",
            )
            shape_rescaled_str = np.array2string(
                shape_rescaled,
                precision=10,
                suppress_small=False,
                separator=", ",
            )
            shape_int_str = np.array2string(
                shape_int,
                separator=", ",
            )
            pcell_str = np.array2string(
                pcell,
                precision=10,
                suppress_small=False,
                separator=", ",
            )
            scell_str = np.array2string(
                scell,
                precision=10,
                suppress_small=False,
                separator=", ",
            )
            raise ValueError(
                f"Atom-count ratio ({n_atoms_scell}/{n_atoms_pcell} = {ratio}) "
                f"does not match |det(shape_int)| = {sc_index}.\n"
                f"shape =\n{shape_str}\n"
                f"shape_rescaled =\n{shape_rescaled_str}\n"
                f"shape_int =\n{shape_int_str}\n"
                f"pcell =\n{pcell_str}\n"
                f"scell =\n{scell_str}"
            )

    return is_shape_int, shape_int, shape_rescaled, shape, sc_index
