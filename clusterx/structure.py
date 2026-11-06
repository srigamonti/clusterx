# Copyright (c) 2015-2025, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from __future__ import annotations

import copy
import operator
import os
import pickle
import warnings
import random
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import atomic_numbers
from ase.db import connect
from ase.io import read, write

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.utils import is_diagonal, ListDict


class Structure(SuperCell):
    """Structure class

    A ``Structure`` object is a :class:`SuperCell <clusterx.super_cell.SuperCell>` object augmented by
    an array of species numbers (or symbols). This array indicates a particular configuration of
    the :class:`SuperCell <clusterx.super_cell.SuperCell>` (a *"decoration"*), in a way which is compatible with
    the sublattices defined in the :class:`SuperCell <clusterx.super_cell.SuperCell>` object.

    The ``Structure`` class inherits from the :class:`SuperCell <clusterx.super_cell.SuperCell>` class.
    Therefore, all methods available to the
    :class:`SuperCell <clusterx.super_cell.SuperCell>` class
    (and to the :class:`ParentLattice <clusterx.parent_lattice.ParentLattice>`
    class, from which :class:`SuperCell <clusterx.super_cell.SuperCell>` inherits) are available
    to the ``Structure`` class. Therefore, look at the documentation of the classes :class:`SuperCell
    <clusterx.super_cell.SuperCell>`
    and :class:`ParentLattice <clusterx.parent_lattice.ParentLattice>` for additional methods for the
    ``Structure`` class.

    **Parameters:**

    ``super_cell``: SuperCell object
         :class:`SuperCell <clusterx.super_cell.SuperCell>` object.
    ``decoration``: list of int (default: ``None``)
        Atomic numbers of the structure. Overriden by ``sigmas`` and ``atomic_symbols`` if not ``None``.
    ``decoration_symbols``: list of strings (default: ``None``)
        Atomic symbols of the structure. Overriden by ``sigmas`` if not ``None``.
    ``sigmas``: list of int (default: ``None``)
        Every site in a supercell is represented by an array of the species that
        can occupy the site. Thus, *taking as reference these arrays*, a possible
        representation of a *decoration* is by indicating the ordinal number of the
        corresponding species. For instance, if the "sites"-based representation of the SuperCell
        is ``{0: [10,11], 1: [25], 2: [12,25]}`` (*), then equivalent structures are obtained with
        ``decoration = [10,25,25]`` or ``decoration_symbols = ["Ne","Mn","Mn"]`` or ``sigmas = [0,0,1]``.
        If not ``None``, ``sigmas`` overrides ``decoration`` and ``decoration_symbols``.
    ``index_tracking``: Bool
    .. todo::
        * check input. If a structure is initialized with a non-allowed substitutional species, and error
        should be raised.

    **Methods:**
    """

    def __init__(
        self,
        super_cell: SuperCell,
        decoration: Sequence | None = None,
        decoration_symbols: Sequence | None = None,
        sigmas: Sequence | None = None,
        track_indices: bool = False,
    ):
        object.__setattr__(self, "scell", super_cell)
        # self.scell = super_cell
        self.sites = super_cell.get_sites()
        self._pbc = super_cell.get_pbc()

        if sigmas is None:
            if decoration_symbols is None:
                self.decor = decoration
            else:
                self.decor = []
                for s in decoration_symbols:
                    self.decor.append(atomic_numbers[s])
                decoration = self.decor

            self.sigmas = np.zeros(len(decoration), dtype=np.int8)
            self.ems = np.zeros(len(decoration), dtype=np.int8)

            for idx, species in enumerate(decoration):
                if species not in self.sites[idx]:
                    raise AttributeError("Error (Structure): decoration not compatible with parent lattice definition.")

            for idx, species in enumerate(decoration):
                self.sigmas[idx] = np.argwhere(np.array(self.sites[idx], dtype=int) == species)[0, 0]
                self.ems[idx] = len(self.sites[idx])
        else:
            self.decor = np.zeros(len(sigmas), dtype=np.int8)
            self.ems = np.zeros(len(sigmas), dtype=np.int8)
            self.sigmas = np.array(sigmas, dtype=np.uint8)
            for idx, sigma in enumerate(sigmas):
                self.decor[idx] = self.sites[idx][sigma]
                self.ems[idx] = len(self.sites[idx])

        self._track_indices = track_indices
        if self._track_indices:
            self._build_index_maps()
        else:
            self._index_maps = None
            self._rng_site_type_weights = None
            self._canonical_site_type_weights = None
        self.precision_positions = 5
        self.p = self.scell.get_transformation()
        self.structure_reduced = None
        self.sigma_grid = None

    def __getattr__(self, name):
        # Only called if normal attribute lookup fails
        try:
            scell = object.__getattribute__(self, "scell")
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
        return getattr(scell, name)

    def __getstate__(self):
        """Return dict for pickling. Keep it versionable."""
        state = self.__dict__.copy()
        # # If you ever add unpicklable fields, remove/transform them here.
        # state.setdefault("_state_version", 1)
        return state

    def __setstate__(self, state):
        """Restore state. Ensure delegate is set before anything else."""
        # Pull out delegate first if present under either name
        scell = state.get("scell", None)
        if scell is not None:
            object.__setattr__(self, "scell", scell)

        # Now restore the rest normally
        self.__dict__.update(state)

        # # Optional: post-load fixes/migrations based on version
        # ver = state.get("_state_version", 1)
        # if ver == 1:
        #     pass  # place migrations here if needed later

    def get_decoration(self):
        decor = np.zeros_like(self.sigmas, dtype=np.int8)
        for idx, sigma in enumerate(self.sigmas):
            decor[idx] = self.sites[idx][sigma]
        return decor

    def _build_index_maps(self):
        """
        Build mappings of which atomic indices are in which sublattice with which sigma.
        self._index_maps[i][j]: ListDict of atomic indices on sublattice i with sigma j.
        """
        index_maps = {}
        sublattices = self.get_sublattice_types()
        # e.g. {0:[14,13], 1:[54,0,32], 2:[11]}

        for sublat_key, species_list in sublattices.items():
            index_maps[sublat_key] = []
            for species in range(len(species_list)):
                index_maps[sublat_key].append(ListDict())

        sigmas = self.get_sigmas()  # occupation for each site
        tags = self.get_tags()  # which sublattice for each type

        for atom_index, (sigma, tag) in enumerate(zip(sigmas, tags)):
            index_maps[tag][sigma].add(atom_index)

        self._index_maps = index_maps
        # e.g. self._index_maps = {0:[[0,2],[1]], 1:[[],[4],[3]], 2:[[5]]}

        self._rng_site_type_weights = []  # used for flip
        self._canonical_site_type_weights = []  # used for swap
        for sublattice, index_lists in self._index_maps.items():
            if len(index_lists) == 1:
                # do not sample for observer sites
                self._rng_site_type_weights.append(0)
                self._canonical_site_type_weights.append(0)
                continue
            index_list_lens = [len(list_dict) for list_dict in index_lists]
            self._rng_site_type_weights.append(sum(index_list_lens))
            # NOTE: an alternative weighting could be by product instead of sum

            # do not sample site types for swaps, where only one of the sigmas
            # is present, as then, no swap is possible
            if np.count_nonzero(index_list_lens) > 1:
                self._canonical_site_type_weights.append(sum(index_list_lens))
            else:
                self._canonical_site_type_weights.append(0)

    def track_indices(self) -> None:
        """Start tracking indices.

        This method needs to be called if track_indices was not called during
        intialization, but an index map is needed for efficient canonical sampling.
        """
        self._track_indices = True
        self._build_index_maps()

    @classmethod
    def from_file(
        cls,
        filepath: str,
        parent_lattice: ParentLattice | None = None
    ) -> Structure:
        """
        Load a Structure object from a serialized file.

        Supports both binary and JSON-based serialization formats. The file is expected
        to have been created using the `Structure.serialize()` method.

        Supported formats:
        - `.pickle`: Full Python object serialization using `pickle`.
        - `.json`  : ASE-compatible JSON database file.
        - `.xyz`   : extended xyz file.

        Parameters
        ----------
        filepath : str
            Path to the serialized file. The format is inferred from the file extension.
        parent_lattice: ParentLattice (optional):
            Provide ParentLattice if it is missing in the .xyz structure file for
            some reason. Overwrites ParentLattice from the structure file.

        Returns
        -------
        Structure
            A reconstructed Structure object based on the contents of the file.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        file_ext = os.path.splitext(filepath)[1].lower()
        if file_ext == ".pickle":
            return cls._load_from_pickle(filepath)
        elif file_ext == ".json":
            return cls._load_from_json(filepath)
        elif file_ext == ".xyz":
            return cls._load_from_xyz(filepath, parent_lattice)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    @classmethod
    def from_sigma_grid(cls, super_cell: SuperCell, sigma_grid: np.ndarray):
        p = super_cell.get_transformation()
        if not is_diagonal(p):
            raise ValueError(
                "Structure cannot be initialized from sigma grid with non-diagonal super cell transformation."
            )
        return cls(super_cell=super_cell, sigmas=sigma_grid.ravel())

    @staticmethod
    def _load_from_pickle(filepath: str) -> Structure:
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @classmethod
    def _load_from_json(cls, filepath: str) -> Structure:
        from ase.db import connect

        db = connect(filepath)
        parent_lattice = ParentLattice.plat_from_dict(db.metadata.get("parent_lattice"))
        tmat = db.get(id=1).data.tmat
        super_cell = SuperCell(parent_lattice=parent_lattice, p=tmat)
        numbers = db.get(id=1).toatoms().get_atomic_numbers()

        return cls(super_cell=super_cell, decoration=numbers)

    @classmethod
    def _load_from_xyz(
        cls,
        filepath: str,
        parent_lattice: ParentLattice | None = None
    ) -> Structure:
        atoms = read(filepath, format="extxyz")
        if parent_lattice is None:
            parent_lattice = ParentLattice.plat_from_dict(atoms.info["parent_lattice"])
        tmat = atoms.info["tmat"]
        super_cell = SuperCell(parent_lattice=parent_lattice, p=tmat)
        numbers = atoms.get_atomic_numbers()
        return cls(super_cell=super_cell, decoration=numbers)

    @staticmethod
    def _decode_ase_dict(d):
        """
        Recursively decode a dictionary exported from an ASE JSON database.

        This method handles the reconstruction of NumPy arrays and ASE Cell objects
        encoded using special keys such as ``"__ndarray__"`` and ``"__ase_objtype__"``.
        It also converts string keys that represent integers back to integers where applicable.

        Parameters
        ----------
        d : dict, list, or primitive
            The object to decode. Can be a nested structure of dictionaries, lists, or
            primitive types.

        Returns
        -------
        Decoded Python object
            A structure where all encoded NumPy arrays and ASE Cell objects are restored.
        """
        if isinstance(d, dict):
            # Handle encoded NumPy arrays
            if "__ndarray__" in d:
                shape, dtype, flat_data = d["__ndarray__"]
                return np.array(flat_data, dtype=dtype).reshape(shape)

            # Handle ASE Cell object
            if d.get("__ase_objtype__") == "cell":
                array_data = Structure._decode_ase_dict(d["array"])
                return Cell(array_data)

            # Recursively decode nested dicts, and convert keys if possible
            result = {}
            for k, v in d.items():
                try:
                    k_int = int(k)
                except (ValueError, TypeError):
                    k_int = k
                result[k_int] = Structure._decode_ase_dict(v)
            return result

        elif isinstance(d, list):
            return [Structure._decode_ase_dict(item) for item in d]

        else:
            return d

    @classmethod
    def from_dict(cls, structure_dict: dict) -> Structure:
        """
        Create a Structure object from a dictionary object.

        This method is used to reconstruct a Structure from a Python dictionary
        that was obtained by loading a JSON file previously created using the
        `Structure.serialize()` method.

        Parameters
        ----------
        structure_dict : dict
            A dictionary containing the deserialized structure data.

        Returns
        -------
        Structure
            An instance of the Structure class initialized with the given data.
        """

        structure_dict = Structure._decode_ase_dict(structure_dict)
        parent_lattice = ParentLattice.plat_from_dict(structure_dict["metadata"]["parent_lattice"])
        tmat = structure_dict[1]["data"]["tmat"]
        numbers = structure_dict[1]["numbers"]

        super_cell = SuperCell(parent_lattice=parent_lattice, p=tmat)

        return cls(super_cell=super_cell, decoration=numbers)

    def _get_roundpos(self):
        return np.around(self.get_positions(), decimals=self.precision_positions)

    def _get_roundcell(self):
        return np.around(self.get_cell(), decimals=self.precision_positions)

    def __repr__(self):
        tokens = []

        tokens.append("pbc={0}".format(self._pbc))

        cell = self._get_roundcell()
        tokens.append("cell={0}".format(cell.tolist()))

        pos = self._get_roundpos()
        tokens.append("pos_car={0}".format(pos.tolist()))

        decor = self.decor
        tokens.append("at_nrs={0}".format(decor))

        return "{0}({1})".format(self.__class__.__name__, ", ".join(tokens))

    def __hash__(self):
        # sp = self._get_roundpos()
        # fingerprint = str(list(zip(sp,self.decor)))
        return hash(self.__repr__())

    def __eq__(self, other):
        spbc = self.pbc
        opbc = other.pbc
        scell = self._get_roundcell()
        ocell = other._get_roundcell()
        spos = self._get_roundpos()
        opos = other._get_roundpos()

        return (
            self.__class__ == other.__class__
            and np.array_equal(spbc, opbc)
            and np.array_equal(scell, ocell)
            and np.array_equal(spos, opos)
            and np.array_equal(self.sigmas, other.sigmas)
        )

    def set_calculator(self, calculator):
        """Set Calculator object for structure."""
        # super(Structure,self).set_calculator(calculator)
        self._calc = calculator

    def get_positions(self, wrap=False, **wrap_kw):
        return super(Structure, self).get_positions(wrap, **wrap_kw)

    def get_sigmas(self):
        """Return decoration array in terms of sigma variables."""
        return self.sigmas

    def init_reduced_structure(self, scell_reduced: SuperCell):
        """Initialize reduced structure, which has the same parent lattice
        as self, but with SuperCell scell_reduced."""
        if not is_diagonal(self.p):
            raise ValueError(
                f"Cannot reduce structure with non-diagonal SuperCell shape {self.p}"
            )
        grid_shape = tuple(
            np.diagonal(self.p).tolist() + [len(self.get_parent_lattice())]
        )
        # copy=False so that sigma_grid is explicitly a view of sigmas
        self.sigma_grid = self.sigmas.reshape(grid_shape, copy=False)
        self.structure_reduced = Structure(
            scell_reduced, sigmas=np.zeros(len(scell_reduced), dtype=int)
        )
        self.p_reduced_diag = np.diagonal(self.structure_reduced.get_transformation())

    def get_degeneracy(self) -> int:
        """Calculate and return the degeneracy of the structure."""
        sigmas = self.get_sigmas()
        unique_transformations = {tuple(sigmas[np.ix_(per)]) for per in self.scell.get_sym_perm()}
        return len(unique_transformations)

    def get_reduced_structure(self, index: int) -> Structure:
        """Return a reduced structure and reduced index.
        Given transformation p, around site index.

        Only local offsets are cached, so additional storage is independent of
        the full supercell size for a fixed reduced shape. The cache is rebuilt
        when either grid shape changes and is created lazily for older pickles.
        """
        if self.structure_reduced is None:
            raise RuntimeError(
                "structure.get_reduced_structure was called before structure.init_reduced_structure"
            )
        grid = self.sigma_grid
        index = operator.index(index)
        if index < 0 or index >= grid.size:
            raise ValueError(f"index {index} is out of bounds for array with size {grid.size}")
        shape = grid.shape
        reduced_shape = tuple(int(p) for p in self.p_reduced_diag)
        key = (shape, reduced_shape)
        if getattr(self, "_reduced_tile_key", None) != key:
            hx, hy, hz = (p // 2 for p in reduced_shape)
            # Match take_tile's centered, odd window, including even input extents.
            dx = np.arange(-hx, hx + 1, dtype=np.intp)[:, None, None]
            dy = np.arange(-hy, hy + 1, dtype=np.intp)[None, :, None]
            dz = np.arange(-hz, hz + 1, dtype=np.intp)[None, None, :]
            center = ((hx * dy.size + hy) * dz.size + hz) * shape[-1]
            self._reduced_tile = (dx, dy, dz, center)
            self._reduced_tile_key = key
        dx, dy, dz, center = self._reduced_tile
        cell, site = divmod(index, shape[-1])
        x, yz = divmod(cell, shape[1] * shape[2])
        y, z = divmod(yz, shape[2])
        # Gather only the final neighborhood. Sequential takes create intermediate
        # arrays that still span the unprocessed axes of the full supercell.
        sigma_grid_reduced = grid[(dx + x) % shape[0], (dy + y) % shape[1], (dz + z) % shape[2], :]
        index_reduced = center + site
        # link current sigmas of structure to sigmas of structure_reduced
        self.structure_reduced.sigmas = sigma_grid_reduced.reshape((-1,), copy=False)
        return self.structure_reduced, index_reduced

    def get_supercell(self):
        """Return SuperCell member of the Structure"""
        return self.scell

    def get_atoms(self, ignore_observer_atoms: bool = False) -> Atoms:
        """Get Atoms object corresponding to the Structure object"""
        if ignore_observer_atoms:
            types = self.get_sublattice_types()
            sublattice_lengths = [len(val) for val in types.values()]
            tags = self.scell.get_tags()
            site_lens = np.array(sublattice_lengths, dtype=int)[tags]
            mask = site_lens > 1
        else:
            tags = self.scell.get_tags()
            site_lens = 2 * np.ones_like(tags, dtype=int)
            mask = site_lens > 1
        tmat = self.get_transformation()
        atoms = Atoms(
            numbers=np.array(self.decor)[mask],
            positions=self.scell.get_positions()[mask],
            tags=tags[mask],
            cell=self.scell.get_cell(),
            pbc=self.scell.get_pbc(),
        )
        atoms.info["tmat"] = tmat
        atoms.new_array("sigmas", self.sigmas[mask], dtype="i")
        return atoms

    def get_atomic_numbers(self) -> np.ndarray:
        """Get decoration array"""
        return self.get_atoms().get_atomic_numbers()

    def get_chemical_symbols(self):
        """Get decoration array"""
        return self.get_atoms().get_chemical_symbols()

    def serialize(self, filepath="structure.json", fmt=None, fname=None):
        """Save the structure to a file in the specified format.

        Parameters
        ----------
        fmt : str, optional
            File format for output (default is "json"). All formats supported by
            ``ase.io.write()`` are accepted. See:
            https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write

        filepath : str, optional
            Path to the output file (default is "structure.json").

        fname : str, optional
            DEPRECATED. Use ``filepath`` instead. This argument will be removed in
            a future version.
        """
        if fname is not None:
            warnings.warn(
                "'fname' is deprecated and will be removed in a future version. Please use 'filepath' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            filepath = fname

        file_ext = os.path.splitext(filepath)[1].lower().lstrip(".")  # remove leading dot
        fmt = fmt or file_ext  # use file extension as format if fmt is not provided

        if fmt == "json":
            db = connect(filepath, type="json", append=False)
            data = {}
            data["tmat"] = self.get_transformation()
            db.write(self.get_atoms(), data=data)
            db.metadata = {"parent_lattice": self._plat.as_dict()}
        elif fmt == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        elif fmt == "xyz":
            atoms = self.get_atoms()
            atoms.info["parent_lattice"] = self._plat.as_dict()
            write(filepath, images=atoms, format="extxyz")
        else:
            write(filepath, images=self.get_atoms(), format=fmt)

        self._fname = filepath

    def get_random_flip(
        self,
        site_type: int | None = None,
        sigma_initial: int | None = None,
        sigma_final: int | None = None,
        rng: random.Random = random,
    ) -> tuple[int, int, int, int]:
        r"""Get a random flip index in given sublattice with given sigmas.

        **Parameters:**

        ``site_type``: integer (required)
            Indicate index of sub-lattice where atom is to be flipped.

        ``sigma_initial``: int or None, optional (default: None)
            Specifies which atomic species (denoted by the occupation variable `sigma`)
            in the sublattice should be flipped.

            - For an n-ary system with `sigma` values in the range [0, ..., n-1]:
            If `sigma_initial = i`, the species `i` is flipped to a randomly chosen
            different species `j ≠ i` from the same range, or to `sigma_final`, if given.

            - If None (default):
            A species `i` is randomly selected and flipped to a randomly chosen
            `j ≠ i`, or to `sigma_final`, if given.

            Examples:
            - Binary system (n = 2, sigma ∈ {0, 1}):
                `sigma_initial = 0` flips to 1; `sigma_flip = 1` flips to 0.

            - Ternary system (n = 3, sigma ∈ {0, 1, 2}):
                `sigma_intial = 1` flips to either 0 or 2, chosen randomly.

        ``rng``: random number generator with .choice method.

        **Return:**
        ``atom_index``: integer
        ``sigma_initial``: integer
        ``sigma_final``: integer
        ``site_type``: integer
        """
        # TODO: for semi-grand-canonical sampling, simplify this function
        if site_type is None:
            # select a random sublattice, weighted by number of sites in sublattice
            # without selecting from observers sites
            site_type = rng.choices(
                list(self._index_maps), weights=self._rng_site_type_weights
            )[0]

        list_dicts = self._index_maps[site_type]
        n_species = len(list_dicts)
        weights = [len(list_dict) for list_dict in list_dicts]

        if sigma_initial is None:
            sigma_initial = rng.choices(range(n_species), weights=weights)[0]

        atom_index = list_dicts[sigma_initial].choice(rng)

        if sigma_final is None:
            weights = [1] * n_species
            weights[sigma_initial] = 0  # avoid choosing same as sigma_initial
            sigma_final = rng.choices(range(n_species), weights=weights)[0]

        return atom_index, sigma_initial, sigma_final, site_type

    def find_site_type(self, i) -> int:
        if self._index_maps is None:
            self._build_index_maps()
        for site_type, list_dicts in self._index_maps.items():
            for list_dict in list_dicts:
                if i in list_dict:
                    return site_type

    def flip(
        self,
        atom_index: int,
        sigma_initial: int,
        sigma_final: int,
        site_type: int | None = None,
    ) -> None:
        """Perform a flip on sublattice site_type from sigma_initial to sigma_final."""
        site_type = self.find_site_type(atom_index) if site_type is None else site_type
        if self._track_indices:
            self._index_maps[site_type][sigma_initial].remove(atom_index)
            self._index_maps[site_type][sigma_final].add(atom_index)
        self.sigmas[atom_index] = sigma_final

    def set_arrays(self, sigmas=None, decoration=None):
        if sigmas is None and decoration is None:
            print("Structure: provide one of sigmas or decoration")

        if sigmas is not None:
            self.sigmas = sigmas
            for idx, sigma in enumerate(sigmas):
                self.decor[idx] = self.sites[idx][sigma]

        if decoration is not None:
            self.decor = decoration

            for idx, species in enumerate(decoration):
                if species not in self.sites[idx]:
                    raise AttributeError("Error (Structure): decoration not compatible with parent lattice definition.")

            for idx, species in enumerate(decoration):
                self.sigmas[idx] = np.argwhere(np.array(self.sites[idx], dtype=int) == species)[0, 0]

        self._build_index_maps()

    def backup_arrays(self):
        """Backup current state of arrays for potential restoration later."""
        self._arrays_backup = {
            "sigmas": copy.deepcopy(self.sigmas),
            "decor": copy.deepcopy(self.decor),
            "_idxs": copy.deepcopy(self._idxs),
            "_comps": copy.deepcopy(self._comps),
        }

    def restore_arrays(self):
        """Restore arrays from backup."""
        if not hasattr(self, "_arrays_backup"):
            raise RuntimeError("No backup found. Call backup_arrays() first.")

        self.sigmas = copy.deepcopy(self._arrays_backup["sigmas"])
        self.decor = copy.deepcopy(self._arrays_backup["decor"])
        self._idxs = copy.deepcopy(self._arrays_backup["_idxs"])
        self._comps = copy.deepcopy(self._arrays_backup["_comps"])

    def get_random_swap_binary(
        self,
        site_type: int | None = None,
        sigma_initial: int = 0,
        sigma_final: int = 1,
        rng: random.Random = random,
    ) -> tuple[int, int, int]:
        r"""Return indices and sigmas of two randomly selected atoms in given sub-lattice.

        **Parameters:**

        ``site_type``: integer (required)
            Indicate index of sub-lattice where atoms are to be swapped.

        ``sigma_initial``: integer (default: 0)
            Sigma of site i before swap

        ``sigma_final``: integer (default: 0)
            Sigma of site i after swap

        **Return:**
        indices i, j and sigmas sigmas[i], sigmas[j] of swap sites (after swap).
        """
        i, s_initial, s_final, site_type = self.get_random_flip(
            site_type, sigma_initial, sigma_final, rng
        )
        res = self.get_random_flip(site_type, s_final, s_initial, rng)
        j = res[0]
        return i, j, site_type

    def get_random_swap(
        self, site_types: list[int] | None, rng: random.Random = random
    ) -> tuple[int, int, int]:
        """Get indices of two randomly selected atoms in randomly selected sub-lattice.

        First, a sublattice from the site_types array is picked at random. Second,
        a pair of species from the selected sublattice are swapped.

        See also :py:meth:`Structure.get_random_swap_binary() <clusterx.structure.Structure.swap_random_binary>`

        **Parameters:**

        ``site_types``: integer array (required)
            Indicate indices of sub-lattices to be considered in the random sub-lattice selection.
        """
        if site_types is None:
            # select a random sublattice, weighted by number of sites in sublattice
            # without selecting from observers sites
            if sum(self._canonical_site_type_weights) == 0:
                raise RuntimeError("Cannot provide random swap with this substitution.")
            site_type = rng.choices(
                list(self._index_maps), weights=self._canonical_site_type_weights
            )[0]
        else:
            # this assumes that site_types with possible swaps are given
            site_type = rng.choice(site_types)

        sigmas_possible = []
        for i, list_dict in enumerate(self._index_maps[site_type]):
            if len(list_dict) > 0:
                sigmas_possible.append(i)
        sigmas_swap = rng.sample(sigmas_possible, k=2)

        return self.get_random_swap_binary(
            site_type, sigma_initial=sigmas_swap[0], sigma_final=sigmas_swap[1], rng=rng
        )

    def swap(self, i: int, j: int, site_type: int | None = None) -> None:
        """Swap sigmas at indices `i` and `j` with matching sublattices `site_type`."""
        si, sj = self.sigmas[i], self.sigmas[j]
        self.flip(i, si, sj, site_type)
        self.flip(j, sj, si, site_type)

    def update_decoration(self, decoration):
        """Update decoration of the structure object

        **Parameters:**

        ``decoration``:
        """
        self.decor = decoration
        self.sigmas = np.zeros(len(decoration), dtype=np.int8)
        for idx, species in enumerate(decoration):
            self.sigmas[idx] = np.argwhere(self.sites[idx] == species)[0, 0]

    def get_fractional_concentrations(self):
        r"""Get fractional concentration of each species on each sublattice

        This function returns a dictionary. The keys of the dictionary, denoted
        with :math:`t`, are the site types that admit substitution (i.e., those returned
        by ``Structure.get_substitutional_tags()``). The values of the dictionary
        are arrays of float, denoted with vectors :math:`\mathbf{c}_t`.
        The coordinates of :math:`\mathbf{c}_t` (:math:`c_{\sigma t}`) are equal to
        :math:`n_{\sigma t}/n_t`, where :math:`\sigma \in [ 0,m-1]`,
        :math:`n_{\sigma t}` is the number of atoms of species :math:`\sigma`
        occupying sites of type :math:`t`, and :math:`n_t` is the total number
        of sites of type :math:`t`. The coordinates of :math:`\mathbf{c}_t` sum
        up to :math:`1`.
        """

        # scell = self.get_supercell()
        # plat = scell.get_parent_lattice()

        concentration = {}

        substutitional_site_types = self.get_substitutional_tags()
        idx_subs = self.get_idx_subs()
        nsites_per_type = self.get_nsites_per_type()
        # sigmas = self.get_sigmas()
        numbers = self.get_atomic_numbers()

        for site_type in substutitional_site_types:
            concentration[int(site_type)] = []
            nsites = nsites_per_type[site_type]
            atom_indices = self.get_atom_indices_for_site_type(site_type)
            for spnr in idx_subs[site_type]:
                n = np.array(numbers[atom_indices]).tolist().count(spnr)
                concentration[int(site_type)].append(n / nsites)

        return concentration

    def get_symbols_per_site_type(self):
        r"""Get fractional concentration of each species on each sublattice

        This function returns a dictionary. The keys of the dictionary, denoted
        with :math:`t`, are the site types that admit substitution (i.e., those returned
        by ``Structure.get_substitutional_tags()``). The values of the dictionary
        are arrays of float, denoted with vectors :math:`\mathbf{c}_t`.
        The coordinates of :math:`\mathbf{c}_t` (:math:`c_{\sigma t}`) are equal to
        :math:`n_{\sigma t}/n_t`, where :math:`\sigma \in [ 0,m-1]`,
        :math:`n_{\sigma t}` is the number of atoms of species :math:`\sigma`
        occupying sites of type :math:`t`, and :math:`n_t` is the total number
        of sites of type :math:`t`. The coordinates of :math:`\mathbf{c}_t` sum
        up to :math:`1`.
        """
        from ase.data import chemical_symbols as cs

        symbols_per_site_type = {}

        substutitional_site_types = self.get_substitutional_tags()
        idx_subs = self.get_idx_subs()

        for site_type in substutitional_site_types:
            symbols_per_site_type[int(site_type)] = []
            for spnr in idx_subs[site_type]:
                symbols_per_site_type[int(site_type)].append(cs[spnr])

        return symbols_per_site_type
