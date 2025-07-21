# Copyright (c) 2015-2025, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import logging
import pickle
import random
import tracemalloc
import warnings
from itertools import combinations, product
from random import sample
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

from clusterx.super_cell import SuperCell
from clusterx.utils import _is_integer_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _filter_property_names(property_name, property_names):
    if property_name and property_names:
        raise ValueError("Only one of property_name or property_names can be provided.")
    if not (property_name or property_names):
        raise ValueError("At least one of property_name or property_names must be provided.")


class DSGenerator:
    """Generation of derivative structures

    A :class:`DSGenerator <clusterx.derivative_structure.DSGenerator>` object is used to generate
    derivative structures from a :class:`parent lattice <clusterx.parent_lattice.ParentLattice>`.

    **Parameters:**

    ``parent_lattice``: :class:`ParentLattice <clusterx.parent_lattice.ParentLattice>` object
        Parent lattice object. Derivative structures originate from this parent lattice.
    """

    def __init__(self, parent_lattice):
        self.plat = parent_lattice

        self.scell_shapes = pd.DataFrame(columns=["shape_id", "shape"])

        self.configurations = pd.DataFrame(columns=["config_id", "sigma", "shape_id"])

        self.masks = {}

        self.property_names = set()

    def serialize(self, filepath):
        """
        Serialize the current instance to a binary file using pickle.

        Args:
            filepath (str): Path to the file where the object will be saved.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, filepath):
        """
        Deserialize an instance of DSGenerator from a binary file using pickle.

        Args:
            filepath (str): Path to the file from which the object will be loaded.

        Returns:
            DSGenerator: An instance of the class loaded from the file.
        """
        with open(filepath, "rb") as f:
            dss = pickle.load(f)

        if not isinstance(dss, cls):
            raise TypeError(f"Expected object of type {cls.__name__}, but got {type(dss).__name__}")

        return dss

    def add_mask(self, mask_name, config_id_list):
        self.masks[mask_name] = config_id_list

    def get_mask(self, mask_name):
        return self.masks[mask_name]

    def get_property_names(self):
        """Return the set of property column names."""
        return self.property_names.copy()

    def get_configurations(self, mask_name=None):
        if mask_name is not None:
            return self.configurations[self.configurations["config_id"] in self.get_mask(mask_name)]
        return self.configurations

    def get_property(self, config_id, property_name):
        """
        Retrieve property value of a configuration by its config_id.

        Parameters:
        config_id (int): The ID of the configuration to retrieve.

        property_name (str):
        the name of the property

        Returns:
        float: The value of the property

        Raises:
        KeyError: If the config_id is not found.
        """
        result = self.configurations[self.configuration["config_id"] == config_id]
        if not result.empty:
            if property_name not in self.configurations.columns:
                raise ValueError(f"Property '{property_name}' does not exist in the configurations DataFrame.")

            return result.iloc[0][property_name]
        raise KeyError(f"Shape ID {config_id} not found.")

    def get_fractional_concentration_binary(self, config_id):
        """
        Retrieve a fractional concentration of configuration by its config_id.

        Parameters:
        config_id (int): The ID of the configuration to retrieve.

        Returns:
        float: The fractional concentration

        Raises:
        KeyError: If the config_id is not found.
        """
        result = self.configurations[self.configurations["config_id"] == config_id]
        if not result.empty:
            column_name = "frconc_binary"

            if column_name not in self.configurations.columns:
                self.add_fractional_concentration_binary()

            result = self.configurations[self.configurations["config_id"] == config_id]
            return result.iloc[0][column_name]
        raise KeyError(f"Shape ID {config_id} not found.")

    def compute_properties(
        self,
        property_name=None,
        property_names=None,
        ase_calculator=None,
        cemodel=None,
        property_solver=None,
        property_solver_kwargs=None,
        linear_reference=None,
        per_formula_unit=False,
    ):
        """Compute properties"""
        _filter_property_names(property_name, property_names)
        scell_cache = {}

        def _get_structure_object(row):
            shape_id = row["shape_id"]
            if shape_id in scell_cache:
                scell = scell_cache[shape_id]
            else:
                scell = SuperCell(self.plat, p=self.get_scell_shape(shape_id))
                scell_cache[shape_id] = scell

            return scell.gen_structure(sigmas=row["sigma"])

        if ase_calculator is not None:

            def _value_func(row):
                struc = _get_structure_object(row)
                ats = struc.get_atoms()
                ats.calc = ase_calculator
                return ats.get_potential_energy()

        elif cemodel is not None:

            def _value_func(row):
                struc = _get_structure_object(row)
                return cemodel.predict(struc)

        elif property_solver is not None:

            def _value_func(row):
                struc = _get_structure_object(row)
                conc = self.get_fractional_concentration_binary(row["config_id"])
                return property_solver(
                    struc,
                    self.get_scell_shape(row["shape_id"]),
                    conc,
                    **property_solver_kwargs,
                )

        # else: # TODO: add this in a later version
        #    raise ValueError("At least one of ase_calculator, cemodel, or property_solver must be provided.")

        if per_formula_unit:

            def _normalize_per_fu(f):
                def _wrapper(row):
                    sc_size = self.get_scell_size(row["shape_id"])
                    return f(row) / sc_size

                return _wrapper

            _value_func = _normalize_per_fu(_value_func)

        if isinstance(linear_reference, list):

            def _subtract_linear_reference(f):
                def _wrapper(row):
                    (x0, p0), (x1, p1) = linear_reference
                    conc = self.get_fractional_concentration_binary(row["config_id"])

                    # straight line through (x0, p0) and (x1, p1)
                    linref = p0 + (p1 - p0) * (conc - x0) / (x1 - x0)
                    return f(row) - linref

                return _wrapper

            _value_func = _subtract_linear_reference(_value_func)

        if property_name:  # single property is computed
            self.set_property_values_iteratively(property_name=property_name, value_func=_value_func)
        elif property_names:  # multiple properties are computed
            self.set_property_values_iteratively(property_names=property_names, value_func=_value_func)

        if linear_reference == "least-squares":
            # get linear reference from linear fit with y = a * conc + b
            if property_name:
                property_names = [property_name]
            self.add_fractional_concentration_binary()
            concentrations = self.configurations["frconc_binary"]
            for column in property_names:
                values = self.configurations[column]
                a, b = np.polyfit(concentrations, values, deg=1)  # might raise warning if only one unique value
                linref = a * concentrations + b
                self.set_property_values_from_array(column, values - linref)
        elif linear_reference == "concentration-endpoints":
            # get linear reference from highest and lowest concentration points
            if property_name:
                property_names = [property_name]
            self.add_fractional_concentration_binary()
            concentrations = self.configurations["frconc_binary"]
            i0, i1 = np.argmin(concentrations), np.argmax(concentrations)
            x0, x1 = concentrations[i0], concentrations[i1]
            for column in property_names:
                values = self.configurations[column]
                p0, p1 = values[i0], values[i1]
                linref = p0 + (p1 - p0) * (concentrations - x0) / (x1 - x0)
                self.set_property_values_from_array(column, values - linref)

    def add_fractional_concentration_binary(self, recompute=False):
        """
        Adds a new column to the configurations DataFrame with the name 'frconc_binary'.

        The values in the column are the computed fractional concentration.
        The parent_lattice definition must correspond to a binary compound.
        """

        if "frconc_binary" in self.configurations.columns and not recompute:
            return

        do_round = True
        round_precision = 6
        # Check if the parent lattice corresponds to a binary compound
        if not self.plat.is_nary(2):
            raise ValueError("The system should be a binary for this function to be used.")

        # Dictionary to cache fractional concentration values
        frconc_dict = {}
        plat = self.plat

        def _compute_frcon(row):
            """
            Computes the fractional concentration for a given row.
            Uses caching for efficiency.
            """
            nsubs = sum(row["sigma"])

            # Create a unique key based on shape_id and nsubs
            key = f'{row["shape_id"]}_{nsubs}'

            # Return cached value if available
            if key in frconc_dict:
                return frconc_dict[key]

            # Compute fractional concentration
            sc_shape = self.get_scell_shape(row["shape_id"])
            scell = SuperCell(self.plat, p=sc_shape)
            struc = scell.gen_structure(sigmas=row["sigma"])

            # Identify the binary sublattice
            sublt = plat.get_sublattice_types()
            sublattice_id = None
            for i, (k, v) in enumerate(sublt.items()):
                if len(v) == 2:  # Binary sublattice
                    sublattice_id = i
                    break

            # Get fractional concentration
            conc = struc.get_fractional_concentrations()[sublattice_id][1]

            if do_round:
                conc = round(conc, round_precision)

            # Cache and return the result
            frconc_dict[key] = conc
            return conc

        self.set_property_values_iteratively(value_func=_compute_frcon, property_name="frconc_binary")

    def add_property_column(self, property_name, default_value=None):
        """
        Adds a new column to the configurations DataFrame with the given property name.

        **Parameters:**

        property_name : str
            The name of the property to be added as a column.
        default_value : optional
            The default value to populate in the new column. If not provided, defaults to None.
        """
        column_name = property_name

        if column_name in self.configurations.columns:
            raise ValueError(f"Column '{column_name}' already exists in the configurations DataFrame.")

        self.configurations[column_name] = default_value

    def set_property_values_iteratively(
        self, value_func, property_name=None, property_names=None, create_if_missing=True, **kwargs
    ):
        """
        Sets property values iteratively using a function to compute values on the fly.

        **Parameters:**

        value_func : callable
            A function that takes a row of the DataFrame and optional keyword arguments, returning the value for the property.
        property_name : str
            The name of the property whose values are to be set.
            Either this or property_names must be provided.
        property_names : list(str)
            The name of the properties whose values are to be set.
            Either this or property_name must be provided.
        create_if_missing : bool, optional
            If True, creates the column if it does not exist. Defaults to True.
        kwargs : dict
            Additional keyword arguments to pass to the value_func.

        **Usage Example:**
        ```python
        def compute_value_with_args(row, multiplier, offset):
            return row["sigma"] * multiplier + offset

        generator.set_property_values_iteratively(
            compute_value_with_args,
            property_name="example_property",
            multiplier=2,
            offset=5
        )
        ```
        """
        _filter_property_names(property_name, property_names)
        if property_name:
            column_names = [property_name]
        elif property_names:
            column_names = property_names

        for column_name in column_names:
            if column_name not in self.configurations.columns:
                if create_if_missing:
                    self.configurations[column_name] = None
                    self.property_names.add(column_name)
                    print(f"Created new column '{column_name}' in the configurations DataFrame.")
                else:
                    raise ValueError(f"Column '{column_name}' does not exist in the configurations DataFrame.")

        if len(column_names) == 1:
            self.configurations[column_names[0]] = self.configurations.apply(
                lambda row: value_func(row, **kwargs), axis=1
            )
        else:
            self.configurations[column_names] = self.configurations.apply(
                lambda row: value_func(row, **kwargs), axis=1, result_type="expand"
            )

    def set_property_values_from_array(self, property_name, values, create_if_missing=True):
        """
        Sets property values using a provided array of values.

        **Parameters:**

        property_name : str
            The name of the property whose values are to be set.
        values : list or array-like
            The values to set in the property column. Must match the number of rows in the DataFrame.
        create_if_missing : bool, optional
            If True, creates the column if it does not exist. Defaults to True.
        """
        column_name = property_name

        if column_name not in self.configurations.columns:
            if create_if_missing:
                self.configurations[column_name] = None
                self.property_names.add(column_name)
                print(f"Created new column '{column_name}' in the configurations DataFrame.")
            else:
                raise ValueError(f"Column '{column_name}' does not exist in the configurations DataFrame.")

        if len(values) != len(self.configurations):
            raise ValueError(
                "Length of values array does not match the number of rows in the configurations DataFrame."
            )

        self.configurations[column_name] = values

    def add_scell_shape(self, shape):
        """
        Add a new cell shape matrix if it does not already exist.

        Parameters:
        shape (np.ndarray): A 3x3 NumPy array representing the cell shape.

        Returns:
        int: The shape_id of the newly added or existing cell shape.
        """

        sc_size = int(round(np.linalg.det(shape)))

        match = self.scell_shapes[self.scell_shapes["shape"].apply(lambda x: np.array_equal(x, shape))]
        if not match.empty:
            return match.iloc[0]["shape_id"]

        new_id = len(self.scell_shapes)
        self.scell_shapes = pd.concat(
            [
                self.scell_shapes,
                pd.DataFrame({"shape_id": [new_id], "shape": [shape], "size": [sc_size]}),
            ],
            ignore_index=True,
        )
        return new_id

    def get_scell_shape(self, shape_id):
        """
        Retrieve a cell shape matrix by its shape_id.

        Parameters:
        shape_id (int): The ID of the cell shape to retrieve.

        Returns:
        np.ndarray: The 3x3 NumPy array representing the cell shape.

        Raises:
        KeyError: If the shape_id is not found.
        """
        result = self.scell_shapes[self.scell_shapes["shape_id"] == shape_id]
        if not result.empty:
            return result.iloc[0]["shape"]
        raise KeyError(f"Shape ID {shape_id} not found.")

    def get_scell_size(self, shape_id):
        """
        Retrieve a cell shape matrix by its shape_id.

        Parameters:
        shape_id (int): The ID of the cell shape to retrieve.

        Returns:
        np.ndarray: The 3x3 NumPy array representing the cell shape.

        Raises:
        KeyError: If the shape_id is not found.
        """
        result = self.scell_shapes[self.scell_shapes["shape_id"] == shape_id]
        if not result.empty:
            return result.iloc[0]["size"]
        raise KeyError(f"Shape ID {shape_id} not found.")

    def add_configuration(self, sigma, shape_id):
        """
        Add a new configuration.

        Parameters:
        sigma (tuple): A tuple of integers representing the atomic configuration.
        shape_id (int): The shape_id linking the configuration to a cell shape.
        """
        new_id = len(self.configurations)
        self.configurations = pd.concat(
            [
                self.configurations,
                pd.DataFrame({"config_id": [new_id], "sigma": [sigma], "shape_id": [shape_id]}),
            ],
            ignore_index=True,
        )

    def generate(
        self,
        supercell_sizes=None,
        shapes_nearest_orthogonal=False,
        num_subs_list=None,
        sc_shape=None,
        sc_shapes=None,
        n_random=None,
        random_state=None,
    ):
        """Generate derivative structures

         **Parameters:**

         ``supercell_sizes``: list or array of int
            List  of integers indicating the number of unit cells in each derivative supercell
        ``shapes_nearest_orthogonal``: boolean or list of integers, default is False.
            If True, for the class of supercells belonging to a hermite normal form (HNF), it will
            select the supercells with angles between cell vectors pairwise closest to orthogonal.
            This search is by default done by transforming the HNFs by 3x3 determinant-1 matrices
            formed with the integers [1,0,-1]. If you pass a list of integers to this argument,
            the passed list will be used insted of the default [1,0,-1]. If False, the original
            HNFs determine the supercell shapes.
         ``num_subs_list``: ragged list of lists or arrays of integers, or list of dict for multilattice case
            every list or array in the ragged list, indicate the number of substituents to be
            considered in a given supercell. The first dimension must coincide with the
            dimension of ``supercell_sizes``.
         ``sc_shape``: 3x3 matrix or None
            if only decorations for a single supercell are wanted, specify it here.
         ``sc_shapes``: list of 3x3 matrix or None
            list of sc_shapes to generate decorations.
        ``n_random``: int or None
            If provided, generate only this number of random configurations per (shape, nsubs).
         ``random_state``: int or None
            If provided, used to seed the random number generators for reproducibility.
        """
        # TODO: make supercell_sizes positional and required argument, as this
        # method does not work without it.
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        for i, num_subs in enumerate(num_subs_list):
            if sc_shape is None and sc_shapes is None:
                sc_size = supercell_sizes[i]

                if shapes_nearest_orthogonal is False:
                    _, unique_sc_shapes = get_unique_supercells(sc_size, self.plat)
                elif shapes_nearest_orthogonal is True:
                    _, unique_sc_shapes = get_unique_supercells_nearest_orthogonal(
                        sc_size, self.plat, elements=[1, 0, -1]
                    )
                elif isinstance(shapes_nearest_orthogonal, list):
                    _, unique_sc_shapes = get_unique_supercells_nearest_orthogonal(
                        n, plat, elements=shapes_nearest_orthogonal
                    )
                else:
                    raise TypeError("Expected False, True, or a list of integers.")

            elif sc_shapes is None:
                sc_size = int(round(np.linalg.det(sc_shape)))
                unique_sc_shapes = [sc_shape]

            else:
                sc_size = int(round(np.linalg.det(sc_shapes[i])))
                unique_sc_shapes = [sc_shapes[i]]

            for idx, t in enumerate(unique_sc_shapes):
                print(
                    f"Start enum of scell shape {idx+1} of {len(unique_sc_shapes)}. Size: {sc_size}, nsubs:{num_subs}"
                )

                for nsubs in num_subs:
                    self.generate_for_shape_nsubs(sc_shape=t, nsubs=nsubs, n_random=n_random)

        print(f"Enumeration complete. Found {len(self.configurations)} unique configurations.\n")

    def generate_for_shape_nsubs(
        self, sc_shape: List[List[int]], nsubs: Optional[Union[int, dict]] = None, n_random: Optional[int] = None
    ):
        """
        Generate derivative structures.

        Parameters:
        -----------
        sc_shape: List[List[int]]
            Shape of the supercell (3x3 list or array of integers).
        nsubs: int
            Number of substitutions.
        n_random: Optional[int]
            If provided, generate only this number of random configurations.

        Returns:
        --------
        List[np.ndarray]
            List of unique sigma configurations.
        """
        self._validate_inputs(sc_shape, nsubs)
        shape_id = self.add_scell_shape(shape=sc_shape)

        logging.info("Start enum for supercell size: %s, nsubs: %s", self.get_scell_size(shape_id), nsubs)

        scell = SuperCell(self.plat, sc_shape)
        natoms = scell.get_natoms()
        symper = scell.get_sym_perm()
        symper_tuples = [tuple(per) for per in symper]
        ssites = scell.get_substitutional_sites()

        recursive = True
        if isinstance(nsubs, int):
            if nsubs > len(ssites):
                logging.error("nsubs cannot exceed the number of substitutional sites.")
                raise ValueError("nsubs cannot exceed the number of substitutional sites.")

            if n_random is None:
                if recursive:
                    n_max = int(scipy.special.binom(len(ssites), nsubs))
                    with tqdm(total=n_max, desc="Finding unique sigmas") as pbar:
                        num_conf = self._generate_all_configurations_recursive(
                            ssites, nsubs, natoms, shape_id, symper_tuples, pbar=pbar
                        )
                else:
                    num_conf = self._generate_all_configurations(ssites, nsubs, natoms, shape_id, symper)

            else:
                num_conf = self._generate_random_configurations(ssites, nsubs, natoms, shape_id, symper, n_random)
        elif isinstance(nsubs, dict):
            tags = scell.get_tags()
            sltypes = scell.get_sublattice_types()

            if n_random is None:
                num_conf = self._generate_all_configurations_multilattice(
                    nsubs, natoms, shape_id, symper, tags, sltypes
                )
            else:
                raise NotImplementedError()
                # num_conf = self._generate_random_configurations(ssites, nsubs, natoms, shape_id, symper, n_random)

        logging.info(
            "Found %s unique configurations of %s substitutions in %s-atom size scell.",
            num_conf,
            nsubs,
            natoms,
        )

    """
    def explore(element, max_depth, current_depth=0, seen=None):
        if seen is None:
            seen = set()
        seen.add(element)

        if current_depth >= max_depth:
            return seen

        children = F(element)
        for child in children:
            seen.update(explore(child, max_depth, current_depth + 1))
        return seen
    """

    def get_child_sigmas(self, sigma: Tuple[int], ssites: List[int], symper_tuples, seen):

        non_zero_indices = {i for i, val in enumerate(sigma) if val != 0}

        children = set()
        for site in ssites:
            if site not in non_zero_indices:
                children.add(sigma[:site] + (1,) + sigma[site + 1 :])

        unique_children = set()
        # full_list = set()
        for child in children:
            if hash(child) not in seen:
                all_hashes = {hash(tuple(child[i] for i in per)) for per in symper_tuples}
                seen |= all_hashes
                unique_children.add(child)

        return unique_children, seen

    def _generate_all_configurations_recursive(
        self,
        ssites,
        nsubs,
        natoms,
        shape_id,
        symper_tuples,
        sigma0=None,
        seen=None,
        current_nsubs=0,
        pbar=None,
        all_configs=None,
    ):

        if sigma0 is None:
            sigma0 = (0,) * natoms

        if seen is None:
            seen = set()

        if all_configs is None:
            all_configs = set()

        sigma0hash = hash(sigma0)
        if sigma0hash not in seen:
            self.add_configuration(sigma=sigma0, shape_id=shape_id)
            all_configs.add(sigma0hash)

        if pbar:
            pbar.update(1)

        if current_nsubs >= nsubs:
            return

        children, seen = self.get_child_sigmas(sigma0, ssites, symper_tuples, seen)
        if pbar:
            pbar.set_postfix(nseen=len(seen), nchildren=len(children), nsub=sum(1 for x in list(children)[0] if x != 0))
        for child in children:
            self._generate_all_configurations_recursive(
                ssites,
                nsubs,
                natoms,
                shape_id,
                symper_tuples,
                sigma0=child,
                seen=seen,
                current_nsubs=current_nsubs + 1,
                pbar=pbar,
                all_configs=all_configs,
            )

        if current_nsubs == 0:
            return len(all_configs)

    def _generate_all_configurations(
        self, ssites, nsubs, natoms, shape_id, symper, cache_quota_bytes=2 * 1024 * 1024 * 1024
    ):
        n_max = int(scipy.special.binom(len(ssites), nsubs))
        logging.info(
            "Max number of configurations for %s substitutions in %s-atom size scell (no sym accounted): %s",
            nsubs,
            natoms,
            n_max,
        )
        logging.info("Starting to find unique configurations...")
        tracemalloc.start()
        num_conf_start = len(self.configurations)

        pbar = tqdm(combinations(ssites, nsubs), total=n_max, desc="Finding unique sigmas")
        full_list = set()

        # Precompute permutations as index tuples
        self._symper_tuples = [tuple(per) for per in symper]

        mode = "fast"
        switched_mode = False

        for con in pbar:
            current_mem, _ = tracemalloc.get_traced_memory()

            sigma_trial = self._create_sigma_array(natoms, con)

            if current_mem > cache_quota_bytes and not switched_mode:
                mode = "slow"
                switched_mode = True
                logging.info("Cache quota exceeded, switching to slow mode.")
                full_list = set()
                for sigma in self.configurations["sigma"]:
                    sigma_hash_canonical = min(hash(tuple(sigma_trial[i] for i in per)) for per in self._symper_tuples)
                    full_list.add(sigma_hash_canonical)

            if mode is "fast":
                sigma_hash = hash(tuple(sigma_trial))

                if sigma_hash not in full_list:
                    all_hashes = {hash(tuple(sigma_trial[i] for i in per)) for per in self._symper_tuples}
                    full_list |= all_hashes
                    self.add_configuration(sigma=sigma_trial, shape_id=shape_id)

            else:
                # Slow path: only store the canonical hash
                sigma_hash_canonical = min(hash(tuple(sigma_trial[i] for i in per)) for per in self._symper_tuples)
                if sigma_hash_canonical not in full_list:
                    full_list.add(sigma_hash_canonical)
                    self.add_configuration(sigma=sigma_trial, shape_id=shape_id)

            # Show memory usage in tqdm postfix
            pbar.set_postfix(mem=self._format_bytes(current_mem), mode=mode, nconf=len(self.configurations["sigma"]))

        tracemalloc.stop()

        return len(self.configurations) - num_conf_start

    def _generate_all_configurations_multilattice(self, nsubs, natoms, shape_id, symper, tags, sublattice_types):

        full_list: Set[Tuple[int, ...]] = set()
        logging.info("Starting to find unique configurations...")

        num_conf_start = len(self.configurations)

        configurations = DSGenerator._generate_multilattice_configurations(natoms, tags, sublattice_types, nsubs)

        for sigma in configurations:
            if tuple(sigma) not in full_list:
                self.add_configuration(sigma=sigma, shape_id=shape_id)
                self._update_full_list(sigma, symper, full_list)

        return len(self.configurations) - num_conf_start

    @staticmethod
    def _generate_multilattice_configurations(n, tags, sublattice_types, nsubs):
        tags = np.array(tags)

        # Map ems key -> list of positions in `tags` that match that ems key
        sublattice_positions = {
            sublattice_type: list(np.where(tags == sublattice_type)[0]) for sublattice_type in nsubs
        }

        # Create generators for each domain

        sublattice_labelings = {
            sublattice_type: DSGenerator._generate_labelings_for_sublattice(
                n, sublattice_positions[sublattice_type], nsubs[sublattice_type]
            )
            for sublattice_type in nsubs
        }

        # Use product of generators
        for combo in product(*sublattice_labelings.values()):
            combined = np.zeros(n, dtype=int)
            for arr in combo:
                combined += arr  # safe because positions are disjoint
            yield combined

    @staticmethod
    def _generate_labelings_for_sublattice(n, sublattice_positions, nsubs):
        """
        Returns generator of labelings for a sublattice

        e.g.
        sublattice_positions = [3,5,6,7,9]
        nsubs = [3,1]

        [(0,[3,5,9]),(1,[7])]

        Parameters:
        -----------
            positions: list
                index of atomic positions to allocate substitutions
            nsubs: list
                number of substitutions of every kind
        """

        def recursive_build(level, used_indices):
            if level == len(nsubs):
                yield []
                return

            available = [i for i in sublattice_positions if i not in used_indices]
            for indices in combinations(available, nsubs[level]):
                new_used = used_indices | set(indices)
                for rest in recursive_build(level + 1, new_used):
                    yield [(level + 1, indices)] + rest

        for assignment in recursive_build(0, set()):
            full_arr = np.zeros(n, dtype=int)
            for value, idxs in assignment:
                for idx in idxs:
                    full_arr[idx] = value
            yield full_arr

    @staticmethod
    def _format_bytes(size_bytes):
        """Convert bytes into human-readable format (KB, MB, GB)."""
        if size_bytes < 1024:
            return f"{size_bytes:.2f} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / 1024**2:.2f} MB"
        else:
            return f"{size_bytes / 1024**3:.2f} GB"

    def _generate_random_configurations(self, ssites, nsubs, natoms, shape_id, symper, n_random):

        full_list: Set[Tuple[int, ...]] = set()
        attempts = 0
        attempts_total = 0
        # max_attempts = n_random * 10  # Limit attempts to avoid infinite loops
        max_attempts = 1000  # Limit attempts to avoid infinite loops

        logging.info("Starting to generate %s random unique configurations...", n_random)

        num_conf_start = len(self.configurations)

        pbar = tqdm(total=n_random, desc="Generating configurations")
        tracemalloc.start()

        while len(self.configurations) - num_conf_start < n_random and attempts < max_attempts:
            con = tuple(sorted(sample(ssites, nsubs)))
            sigma = self._create_sigma_array(natoms, con)

            is_new = not any(tuple(sigma.take(per, axis=0)) in full_list for per in symper)

            if is_new:
                self.add_configuration(sigma=sigma, shape_id=shape_id)
                full_list.add(tuple(sigma))
                pbar.update(1)

                # Snapshot memory after update
                current, peak = tracemalloc.get_traced_memory()
                pbar.set_postfix(mem=self._format_bytes(current))
                attempts = 0

            attempts += 1
            attempts_total += 1

        pbar.close()
        tracemalloc.stop()

        num_conf = len(self.configurations) - num_conf_start

        if num_conf < n_random:
            logging.info(
                "%s unique configurations could be generated after %s attempts. Desired number: %s",
                num_conf,
                attempts_total,
                n_random,
            )
        return num_conf

    def _generate_random_configurations2(self, ssites, nsubs, natoms, shape_id, symper, n_random):

        full_list: Set[Tuple[int, ...]] = set()
        attempts = 0
        attempts_total = 0
        # max_attempts = n_random * 10  # Limit attempts to avoid infinite loops
        max_attempts = 1000  # Limit attempts to avoid infinite loops

        logging.info("Starting to generate %s random unique configurations...", n_random)

        num_conf_start = len(self.configurations)

        pbar = tqdm(total=n_random, desc="Generating configurations")
        tracemalloc.start()

        while len(self.configurations) - num_conf_start < n_random and attempts < max_attempts:
            con = tuple(sorted(sample(ssites, nsubs)))
            sigma = self._create_sigma_array(natoms, con)

            if tuple(sigma) not in full_list:
                self.add_configuration(sigma=sigma, shape_id=shape_id)
                self._update_full_list(sigma, symper, full_list)
                attempts = 0
                pbar.update(1)

                # Snapshot memory after update
                current, peak = tracemalloc.get_traced_memory()
                pbar.set_postfix(mem=self._format_bytes(current))

            attempts += 1
            attempts_total += 1

        pbar.close()
        tracemalloc.stop()

        num_conf = len(self.configurations) - num_conf_start

        if num_conf < n_random:
            logging.info(
                "%s unique configurations could be generated after %s attempts. Desired number: %s",
                num_conf,
                attempts_total,
                n_random,
            )
        return num_conf

    def _validate_inputs(self, sc_shape, nsubs):
        if not (
            isinstance(sc_shape, (list, np.ndarray)) and len(sc_shape) == 3 and all(len(row) == 3 for row in sc_shape)
        ):
            raise ValueError("sc_shape must be a 3x3 list or array of integers.")

        if not (isinstance(nsubs, dict) or (isinstance(nsubs, int) and nsubs >= 0)):
            raise ValueError("nsubs must be a dict or a non-negative integer.")

    def _create_sigma_array(
        self, natoms: int, con: Tuple[int], sigma: int = 1, sigmas: np.ndarray = None
    ) -> np.ndarray:
        """Create a sigma array from a combination."""
        if sigmas is None:
            sigmas = np.zeros(natoms, dtype="int")

        np.put(sigmas, con, sigma)
        return sigmas

    def _update_full_list(self, sigmas, symper, full_list):
        """Update the set of unique configurations."""
        for sigmas_perm in [sigmas[np.ix_(per)] for per in symper]:
            full_list.add(tuple(sigmas_perm))


def _divisors(n):
    # get factors and their counts
    # Mainly taken from https://stackoverflow.com/a/37058745
    factors = {}
    nn = n
    i = 2
    while i * i <= nn:
        while nn % i == 0:
            if i not in factors:
                factors[i] = 0
            factors[i] += 1
            nn //= i
        i += 1
    if nn > 1:
        factors[nn] = 1

    primes = list(factors.keys())

    # generates factors from primes[k:] subset
    def generate(k):
        if k == len(primes):
            yield 1
        else:
            rest = generate(k + 1)
            prime = primes[k]
            for factor in rest:
                prime_to_i = 1
                # prime_to_i iterates prime**i values, i being all possible exponents
                for _ in range(factors[prime] + 1):
                    yield factor * prime_to_i
                    prime_to_i *= prime

    # in python3, `yield from generate(0)` would also work
    # for factor in generate(0):
    #    yield factor

    r = []
    for factor in generate(0):
        r.append(factor)

    return sorted(r)


def get_unique_supercells(n, parent_lattice):
    """Find full list of unique supercells of index n.

    Following Ref.[1], the complete set of symmetrically inequivalent HNFs of
    index ``n``, for a given ``parent_lattice``, is determined and returned.

    [1] Gus L. W. Hart and Rodney W. Forcade *Phys. Rev. B*
    **80**, 014120 (2009).

    **Parameters:**

    ``n``: integer
        index of the supercells, i.e., the number of atoms in the supercells is
        ``n*parent_lattice.get_natoms()``.
    ``parent_lattice``: ParentLattice object
        The parent lattice

    **Returns:** two arrays containing 3x3 matrices. The matrices ``S`` of the
    first array contain the cartesian coordinates of the supercell vectors (row
    wise), while the matrices ``H`` in the second array are the (integer)
    transormation matrices with respect to the parent lattice ``U``. That is,
    :math:`S=HU`

    **Example:**
    In the following example, all the supercells of index 4 for the FCC lattice
    are found. The supercells are stored in the file
    ``unique_supercells-fcc.json`` for visualization with the command
    ``$>ase gui unique_supercells-fcc.json`` ::

        from clusterx import utils
        from clusterx.parent_lattice import ParentLattice
        from clusterx.structures_set import StructuresSet
        from clusterx.super_cell import SuperCell
        from clusterx.structure import Structure
        from ase.data import atomic_numbers as an
        from ase import Atoms
        import numpy as np

        a=3
        cell = np.array([[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])
        positions = np.array([[0,0,0]])
        sites = [[an["Cu"],an["Au"]]]
        pris_fcc = Atoms(cell=cell*a,positions=positions*a)

        pl = ParentLattice(pris_fcc,sites=sites)

        unique_scs, unique_sc_shapes = utils.get_unique_supercells(4,pl)

        sset = StructuresSet(pl,filename="unique_supercells-fcc.json")
        for t in unique_sc_shapes:
            scell = SuperCell(pl,t)
            sset.add_structure(Structure(scell,scell.get_atomic_numbers()),write_to_db = True)

    The generated structures are the same as those found in Fig. 2 and Table IV
    of Phys. Rev. B 77, 224115 2008.

    The next example, shows a case of reduced dimensionality, that of a 2D
    square lattice::

        a=3.1
        cell = np.array([[1,0,0],[0,1,0],[0,0,1]])
        positions = np.array([[0,0,0]])
        sites = [[12,13]]
        pris = Atoms(cell=cell*a, positions=positions*a)

        pl = ParentLattice(pris, sites=sites, pbc=(1,1,0))

        unique_scs, unique_sc_shapes = utils.get_unique_supercells(4,pl)

        sset = StructuresSet(pl,filename="test_get_unique_supercells-square_lattice.json")
        for t in unique_sc_shapes:
            scell = SuperCell(pl,t)
            sset.add_structure(Structure(scell,scell.get_atomic_numbers()),write_to_db = True)

        #isok0 = len(unique_scs) == 4 and
        print("n: ",len(unique_scs))
        print("SCS: ", unique_scs)
        print("TRA: ", unique_sc_shapes)

    The resulting supercells in this example correspond to Fig. 1 of
    Computational Materials Science 59 (2012) 101–107

    """
    pl_cell = parent_lattice.get_cell()

    hnfs = get_HNFs(n, pbc=parent_lattice.get_pbc())

    all_scs = []
    for hnf in hnfs:
        all_scs.append(np.dot(hnf, pl_cell))

    n_scs = len(all_scs)
    unique_scs = []
    unique_sc_shapes = []
    sc_sg, sc_sym = parent_lattice.get_sym()  # Scaled to parent_lattice
    nexts = np.asarray(np.arange(n_scs))
    unique_scs.append(all_scs[0])
    unique_sc_shapes.append(hnfs[0])

    while len(nexts) > 1:
        i = nexts[0]
        the_list = nexts[1:]
        nexts = []
        bi = all_scs[i]
        for j in the_list:
            bj = all_scs[j]
            j_is_next = True
            for r in sc_sym["rotations"]:
                rr = np.dot(
                    pl_cell, np.dot(r, np.linalg.inv(pl_cell))
                )  # Rotations are in lattice coordinates, so we have to transorm them to cartesian.
                m = np.around(np.dot(np.linalg.inv(bi.T), np.dot(rr, bj.T)), 5)
                if _is_integer_matrix(m):
                    j_is_next = False
                    break

            if j_is_next:
                nexts.append(j)

        if len(nexts) > 0:
            unique_scs.append(all_scs[nexts[0]])
            unique_sc_shapes.append(hnfs[nexts[0]])

    return unique_scs, unique_sc_shapes


def get_HNFs(n, pbc=(1, 1, 1)):
    """Return complete set of Hermite normal form (HNF) :math:`3x3` matrices
    of index ``n``.

    The algorithm here is based on Equation 1 of Gus L. W. Hart and Rodney W.
    Forcade, *Phys. Rev. B* **77**, 224115 (2008).

    **Parameters:**

    ``n``: integer
        index of the HNF matrices.
    ``pbc``: three bool
        Periodic boundary conditions flags. Examples:
        (1, 1, 0), (True, False, False). Default value: (1,1,1)
    """

    _hnfs = []
    for a in _divisors(n):
        for c in _divisors(int(n / a)):
            f = int(n / (a * c))
            for b in range(c):
                for d in range(f):
                    for e in range(f):
                        hnf = []
                        hnf.append([a, 0, 0])
                        hnf.append([b, c, 0])
                        hnf.append([d, e, f])
                        _hnfs.append(np.array(hnf).T)

    hnfs = []
    for hnf in _hnfs:
        include = True
        for i, bc in enumerate(pbc):
            if not bc and (hnf[i] != np.identity(3, dtype="int")[i]).any():
                include = False
                break
        if include:
            hnfs.append(hnf)

    return hnfs


def _get_minimal_sc_shape(h, all_matrices=None, cell=None):

    def minimum_key(x):
        return max(_get_normalized_scalar_products(np.dot(x, cell)))

    sc_shapes = [np.dot(np.reshape(mat, (3, 3)), h) for mat in all_matrices]
    minimal = min(sc_shapes, key=minimum_key)
    return minimal


def _get_normalized_scalar_products(s: np.ndarray):
    """
    For a matrix of column vectors, return the normalized scalar products.

    **Parameters:**

    ``s``: numpy.ndarray
        matrix of transformed cell vectors as columns

    **Returns:**

    Normalized scalar products between unique vector pairs
    """
    p_ij = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        S_i = s.T[i]
        S_j = s.T[j]
        denominator = np.linalg.norm(S_i) * np.linalg.norm(S_j)
        nominator = abs(np.dot(S_i, S_j))
        p_ij.append(nominator / denominator)
    p_ij = np.array(p_ij)
    return p_ij


def get_unique_supercells_small_angles(n, parent_lattice: object, elements: list):
    warnings.warn(
        "'get_unique_supercells_small_angles' is deprecated and will be removed in a future release. "
        "Please use 'get_unique_supercells_nearest_orthogonal' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_unique_supercells_nearest_orthogonal(n, parent_lattice, elements)


def get_unique_supercells_nearest_orthogonal(n, parent_lattice: object, elements: list):
    """
    Return all unique supercells with pairwise lattice vectors angles closest to orthogonal.
    Transformation of those supercells is done by unimodal matrices constructed
    with integer matrix elements given by the paramter ``elements``

    **Parameters:**

    ``elements``: list[int]
        transformation vector elements

        example: [-3,-2,-1,0,1,2,3]

        The algorithm will search all combinations of 3x3 matrices composed of
        those ``elements`` and filter those with determinant 1.

    """

    import multiprocessing
    from functools import partial
    from itertools import product

    from clusterx.super_cell import SuperCell  # needed by make_supercell

    _, harray = get_unique_supercells(n, parent_lattice)

    parent_lattice_cell = parent_lattice.get_cell().array.T

    all_matrices = list(
        filter(
            lambda x: abs(np.linalg.det(np.reshape(x, (3, 3)))) == 1,
            product(elements, repeat=9),
        )
    )

    with multiprocessing.Pool() as pool:
        small_angle_sc_shapes = pool.map(
            partial(
                _get_minimal_sc_shape,
                all_matrices=all_matrices,
                cell=parent_lattice_cell,
            ),
            harray,
        )

    return [SuperCell(parent_lattice, p) for p in small_angle_sc_shapes], small_angle_sc_shapes
