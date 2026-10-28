# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import importlib.util
import os
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plac
from ase.calculators.calculator import Calculator

from clusterx.derivative_structures import DSGenerator
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.utils import normalize_nsubs_list, normalize_shape_input
from clusterx.visualization import (
    plot_predictions_vs_target,
    plot_property_vs_concentration,
)

commands = ["generate_derivative_structures"]


@plac.annotations(
    sc_sizes=("List of supercell sizes.", "option", "scsi", list),
    shapes_nearest_orthogonal=(
        "Generate supercell shapes with angles closest to orthogonal",
        "flag",
        "sno",
        bool,
    ),
    nsubs_list=(
        "List of lists indicating number of substitutions for each supercell.",
        "option",
        "nsl",
        list,
    ),
    sset_filepath=("Path to a serialized StructuresSet object.", "option", "ssfp", str),
    sset_gss_filepath=(
        "Path to a serialized GSS StructuresSet object.",
        "option",
        "ssgfp",
        str,
    ),
    model_filepath=("Filepath of a serialized CE model object.", "option", "mfp", str),
    plat_filepath=(
        "Filepath of a serialized ParentLattice object.",
        "option",
        "plfp",
        str,
    ),
    dss_filepath=(
        "Filepath to serialize or retrieve derivative structures.",
        "option",
        "dssfp",
        str,
    ),
    dss_filepaths=(
        "List of filepaths to existing derivative structures for task=merge_dsses.",
        "option",
        "dssfps",
        list,
    ),
    merge_mask_names=(
        "For task=merge_dsses: list of mask names, one per source in dss_filepaths. "
        "Use None for a source to include all its configurations.",
        "option",
        "mmns",
        list,
    ),
    merged_dss_filepath=(
        "Filepath to serialize or retrieve merged derivative structures.",
        "option",
        "mdssfp",
        str,
    ),
    property_name=(
        "Name of the property to request from the structures set.",
        "option",
        "plab",
        str,
    ),
    property_names=(
        "Names of the property to request from the structures set.",
        "option",
        "plabs",
        list,
    ),
    property_solver=(
        "Dictionary of parameters for the property solver.",
        "option",
        "psol",
        dict,
    ),
    sc_shape=("3x3 integer matrix to specify supercell shape.", "option", "scsh", list),
    per_formula_unit=("Flag: compute per formula unit.", "flag", "pfu", bool),
    linear_reference=(
        "Linear reference for property correction.",
        "option",
        "lref",
        list,
    ),
    mask_name=("Name of the mask for applicable tasks.", "option", "mn", str),
    n_lowest=("Number of lowest-energy structures to include.", "option", None, int),
    n_random=("Task-dependent number of random structures.", "option", None, int),
    random_state=("Seed for random number generators.", "option", None, int),
    task=("Task to perform.", "option", "task", str),
    skip_existing=(
        "when adding binary complements, skip those that already exist in the DSS instead of raising an error",
        "flag",
        "skip",
        bool,
    ),
    non_recursive=(
        "Use slow non-recursive method for finding derivative structures",
        "flag",
        "rec",
        bool,
    ),
    y_axis_label=("Label for the y-axis of the plot.", "option", "y", str),
    ymin=("Minimum y-value for the plot axis.", "option", None, float),
    ymax=("Maximum y-value for the plot axis.", "option", None, float),
    show_plot=("If True, show the plot in a new window.", "option", "sp", bool),
    save_plot=("If True, save the plot to file.", "option", "svp", bool),
    store_ensemble_predictions=(
        "If True, for task compute_property_with_ce_model, and if the model in model_filepath is a bagging regressor.",
        "option",
        "sep",
        bool,
    ),
    plot_fname=("Filepath to save the plot at.", "option", "pf", str),
)
def generate_derivative_structures(
    sc_sizes: Optional[List[int]] = None,
    shapes_nearest_orthogonal: Union[bool, List[int]] = False,
    nsubs_list: Optional[Union[int, List[int], List[List[int]]]] = None,
    sset_filepath: Optional[str] = None,
    sset_gss_filepath: Optional[str] = None,
    model_filepath: Optional[str] = None,
    plat_filepath: Optional[str] = None,
    dss_filepath: Optional[str] = "dss.pickle",
    dss_filepaths: Optional[List[str]] = None,
    merge_mask_names: Optional[List[Optional[str]]] = None,
    merged_dss_filepath: Optional[str] = None,
    property_name: Optional[str] = None,
    property_names: Optional[List[str]] = None,
    property_solver: Optional[dict] = None,
    sc_shape: Optional[Union[int, List[int], List[List[int]]]] = None,
    per_formula_unit: bool = False,
    linear_reference: Optional[Union[List[List[float]], List[dict], str]] = None,
    mask_name: Optional[Union[str, List[Optional[str]]]] = None,
    n_lowest: Optional[int] = 1,
    n_random: Optional[int] = 0,
    random_state: Optional[int] = None,
    task: str = "do_full_enumeration",
    y_axis_label: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    show_plot: bool = True,
    save_plot: bool = True,
    store_ensemble_predictions: bool = False,
    plot_fname: str = "property_vs_concentration.png",
    colors=None,
    markers=None,
    sizes=None,
    skip_existing=False,
    non_recursive=False,
):
    """
    Generate derivative structures and (optionally) compute/plot properties.

    This command wraps several workflows around :class:`clusterx.derivative_structures.DSGenerator`,
    :class:`clusterx.parent_lattice.ParentLattice`, and :class:`clusterx.model.Model`.
    The behavior is controlled by ``task`` (see **Tasks** below).

    By default, results are serialized to ``dss.pickle`` (a pickled ``DSGenerator`` with
    its generated configurations and masks). Some tasks may also produce plots or update
    properties inside the serialized object.

    **Typical flow**

    1. Enumerate or randomly sample derivative structures from a parent lattice.
    2. Compute a property for every configuration (via CE model, calculator, or a custom solver).
    3. Post-process (e.g., mark lowest values per concentration) and/or visualize.

    Parameters
    ----------
    sc_sizes : list[int], optional
        Supercell sizes to consider during enumeration/random generation.
    shapes_nearest_orthogonal : bool or list[int], default ``False``
        If ``True`` (or a list of shape IDs), prefer supercell shapes with angles closest to orthogonal.
    nsubs_list : int or list[int] or list[list[int]], optional
        Number(s) of substitutions per supercell (can be a single int, a list, or a list per shape).
    sset_filepath : str, optional
        Path to a serialized :class:`clusterx.structures_set.StructuresSet` (needed by some plotting tasks).
    model_filepath : str, optional
        Path to a serialized :class:`clusterx.model.Model` used for CE predictions.
    plat_filepath : str, optional
        Path to a serialized :class:`clusterx.parent_lattice.ParentLattice` (required by enumeration/random tasks).
    dss_filepath : str, default ``"dss.pickle"``
        Output/input pickle filepath for the :class:`clusterx.derivative_structures.DSGenerator`.
    dss_filepaths : list[str], optional
        Input filepaths for ``task="merge_dsses"``.
    merge_mask_names : list[str | None], optional
        For ``task="merge_dsses"``, mask names to apply per input source
        (same length as ``dss_filepaths``). Use ``None`` to include all configurations
        for that source.
    property_name : str, optional
        Single property name to compute/plot. Mutually exclusive with ``property_names``.
    property_names : list[str], optional
        Multiple property names to compute/plot. Mutually exclusive with ``property_name``.
    property_solver : dict, optional
        Configuration for the custom solver used by
        ``task="compute_property_with_custom_solver"``. Provide ``filename``
        (path to a Python module) and exactly one of:

        - ``function``: name of a module-level function to call.
        - ``classname``: name of a solver class. Optionally provide ``method``
          (defaults to ``"compute_property"``) to select a method on its
          instance.

        Optionally provide ``kwargs`` (a dictionary passed to the selected
        callable).
    sc_shape : int or list[int] or list[list[int]], optional
        Supercell shape specification. May be a scalar multiplier or an explicit 3×3 integer matrix.
    per_formula_unit : bool, default ``False``
        If ``True``, compute/scale properties per formula unit.
    linear_reference : list[list[float]] or list[dict] or str, optional
        Linear-reference definition, active for
        ``task="compute_property_with_ce_model"`` (or ``"compute_property_ce"``)
        and ``task="compute_property_with_custom_solver"``. The linear reference
        is subtracted to the computed property.
        Pass two concentration/property points as ``[[x0, p0], [x1, p1]]`` or
        ``[{'x': x0, 'y': p0}, {'x': x1, 'y': p1}]``, or use
        ``"least-squares"`` / ``"concentration-endpoints"`` to define a reference by fitting
        the computed values. ``None`` disables the linear-reference.
        The behavior and constraints are documented in
        :meth:`clusterx.derivative_structures.DSGenerator.compute_properties`.
    mask_name : str or list[str | None], optional
        Name of a mask to write/read within the serialized DSS object for downstream
        filtering. For ``task=merge_dsses``, a single string applies the same mask to
        all sources; a list is interpreted as per-source masks.
    n_lowest : int, default ``1``
        Count of lowest-value configurations per concentration to mark (used by
        ``"mark_lowest_property_per_concentration"`` / ``"mark_lowest_and_random_properties_per_concentration"``).
    n_random : int, default ``0``
        Task-dependent number of random configurations:

        - For ``task="random"``, the number to generate for each
          (supercell shape, substitution count) pair.
        - For ``task="mark_lowest_and_random_properties_per_concentration"``,
          the number to sample per concentration and add to the mask alongside
          the ``n_lowest`` configurations. Duplicates are removed, so fewer
          than ``n_random`` additional configurations may be marked.
    random_state : int, optional
        Seed for random selection/generation.
    task : str, default ``"do_full_enumeration"``
        Selects the workflow to run. See **Tasks** below.
    plotdata_filepath : str, optional
        When plotting multiple series, save the plotted ``.npz`` data to this path.
    colors, markers, sizes : list, optional
        Styling for multi-series scatter plots (length should match the number of series).
    non_recursive : bool, default ``False``
        If ``True``, use the slower non-recursive enumeration method.

    CLI flags (plac)
    ----------------
    Many parameters are exposed via short options when used as a CLI command:

    - ``--scsi`` → ``sc_sizes``
    - ``--sno`` → ``shapes_nearest_orthogonal`` (flag)
    - ``--nsl`` → ``nsubs_list``
    - ``--ssfp`` → ``sset_filepath``
    - ``--mfp`` → ``model_filepath``
    - ``--plfp`` → ``plat_filepath``
    - ``--dssfp`` → ``dss_filepath``
    - ``--dssfps`` → ``dss_filepaths``
    - ``--mmns`` → ``merge_mask_names``
    - ``--plab`` → ``property_name``
    - ``--plabs`` → ``property_names``
    - ``--psol`` → ``property_solver``
    - ``--scsh`` → ``sc_shape``
    - ``--pfu`` → ``per_formula_unit`` (flag)
    - ``--lref`` → ``linear_reference``
    - ``--mn`` → ``mask_name``
    - ``--task`` → ``task``
    - ``--rec`` → sets ``non_recursive=True`` (use non-recursive enumeration)

    Tasks
    -----
    ``"random"``
        Randomly generate up to ``n_random`` unique configurations per
        (supercell shape, substitution count) under the given constraints.
        Writes the resulting DSS to ``dss_filepath``.
    ``"do_full_enumeration"`` or ``"find_derivative_structures"``
        Exhaustively enumerate derivative structures for the given lattice/sizes/shapes.
        Writes the resulting DSS to ``dss_filepath``. Honor ``non_recursive``.
    ``"compute_property_with_ce_model"`` or ``"compute_property_ce"``
        Compute ``property_name`` (or ``property_names``) for all configurations in ``dss_filepath``
        using the CE model from ``model_filepath``. Supports ``per_formula_unit`` and ``linear_reference``.
    ``"compute_property_with_custom_solver"``
        Load and call the module-level function or class method selected by
        ``property_solver`` for every configuration via
        :meth:`clusterx.derivative_structures.DSGenerator.compute_properties`.
        The selected callable receives the generated clusterX structure, its
        3×3 supercell shape, the binary fractional concentration, and the
        configuration's pandas Series as ``row``; optional
        ``property_solver["kwargs"]`` are forwarded as keyword arguments.
        Return one scalar when ``property_name`` is set, or an ordered sequence
        with one value per entry in ``property_names``. The CLI passes these
        arguments through to ``compute_properties``; see its docstring for
        details on return values, ``per_formula_unit``, and ``linear_reference``.
    ``"plot_property_vs_concentration"`` / ``"plot_property"``
        Scatter plot property vs fractional concentration. If ``sset_filepath`` and
        ``model_filepath`` are given, overlays CE predictions.
    ``"plot_predictions_vs_target"``
        Quick diagnostic scatter: target values vs CE predictions for ``property_name`` using
        ``sset_filepath`` and ``model_filepath``.
    ``"plot_property_vs_concentration2"`` / ``"plot_property_vs_concentration3"``
        Alternative interactive visualizations (Bokeh / Plotly).
    ``"mark_lowest_property_per_concentration"``
        Create a mask with the configuration of minimal ``property_name`` for each concentration bin.
    ``"mark_lowest_and_random_properties_per_concentration"``
        Create a mask with the ``n_lowest`` best plus up to ``n_random`` randomly
        sampled configurations per concentration.
    ``"to_sset"`` / ``"convert_to_sset"``
        Convert the stored configurations (optionally filtered by ``mask_name``) into a
        :class:`clusterx.structures_set.StructuresSet` written to ``sset_filepath``.

    Returns
    -------
    None
        Results are persisted to ``dss_filepath`` and/or ``sset_filepath``; some tasks display plots.

    Notes
    -----
    - Exactly one of ``property_name`` or ``property_names`` should be provided when computing/plotting properties.
    - The serialized DSS object contains: configurations (DataFrame), masks (dict of config_id arrays), and supercell shape metadata used by conversion/plotting tasks.

    Examples
    --------
    **1) Create a set of random structures and serialize them**

    .. code-block:: toml

        [generate_derivative_structures]
        task = "random"
        plat_filepath = "plat.json"
        sc_shape = 2
        nsubs_list = [0, 2, 4, 6, 8]
        n_random = 5
        random_state = 1
        dss_filepath = "random_structures.pickle"

    **2) Define a custom solver function**

    Save a module-level function in a Python file, for example
    ``my_custom_solvers.py``. Functions follow the callable interface
    documented by :meth:`clusterx.derivative_structures.DSGenerator.compute_properties`:

    .. code-block:: python

        def compute_energy(structure, shape, concentration, row, scale=1.0):
            # Replace this example calculation with the desired property.
            return scale * concentration

    **3) Run a custom solver function**

    .. code-block:: toml

        [generate_derivative_structures]
        task = "compute_property_with_custom_solver"
        property_solver = { filename = "path/to/my_custom_solvers.py",
                            function = "compute_energy",
                            kwargs = { scale = 2.0 } }
        dss_filepath = "random_structures.pickle"
        property_name = "energy"

    **4) Select a method from a solver class**

    Use a class when you want to group related solvers or share instance state.
    The class is instantiated without arguments. The ``method`` key selects
    which instance method is called; if omitted, it defaults to
    ``"compute_property"``.

    .. code-block:: python

        class MySolvers:
            def compute_property_1(
                self, structure, shape, concentration, row, scale=1.0, offset=0.0
            ):
                return scale * concentration + offset

            def compute_property_2(
                self, structure, shape, concentration, row, normal=3.0, parity=2.0
            ):
                return row["property"] * concentration / normal + parity

    .. code-block:: toml

        [generate_derivative_structures]
        task = "compute_property_with_custom_solver"
        property_solver = { filename = "path/to/my_custom_solvers.py",
                            classname = "MySolvers",
                            method = "compute_property_2",
                            kwargs = { normal = 3.0, parity = 2.0 } }
        dss_filepath = "random_structures.pickle"
        property_name = "property_2"

    **5) Merge multiple DSS files with optional per-source masks**

    .. code-block:: toml

        [generate_derivative_structures]
        task = "merge_dsses"
        dss_filepaths = ["a.pickle", "b.pickle", "c.pickle"]
        merged_dss_filepath = "merged.pickle"
        # One entry per source; None means include all configurations from that source
        merge_mask_names = ["lowest", "candidate_pool", "none"]

    You can also provide a single ``mask_name`` to apply the same mask to all
    sources during merge.

    For the custom solver callable contract, including multi-property return
    values, see :meth:`clusterx.derivative_structures.DSGenerator.compute_properties`.
    """

    match task:
        # Generate random structures
        case "random":
            plat = ParentLattice(filepath=plat_filepath)
            dsgen = DSGenerator(plat)
            sc_shape = normalize_shape_input(sc_shape)
            nsubs_list = normalize_nsubs_list(nsubs_list)
            dsgen.generate(
                num_subs_list=nsubs_list,
                supercell_sizes=sc_sizes,
                shapes_nearest_orthogonal=shapes_nearest_orthogonal,
                sc_shape=sc_shape,
                n_random=n_random,
                random_state=random_state,
            )

            with open(dss_filepath, "wb") as f:
                pickle.dump(dsgen, f)

        # Find derivative structures
        case "do_full_enumeration" | "find_derivative_structures":
            plat = ParentLattice(filepath=plat_filepath)
            sc_shape = normalize_shape_input(sc_shape)
            nsubs_list = normalize_nsubs_list(nsubs_list)

            _do_full_enumeration(
                plat,
                nsubs_list=nsubs_list,
                sc_sizes=sc_sizes,
                shapes_nearest_orthogonal=shapes_nearest_orthogonal,
                sc_shape=sc_shape,
                dss_filepath=dss_filepath,
                recursive=not non_recursive,
            )

        case "add_binary_complements":
            print("Loading DSS from file...")
            dss = DSGenerator.from_file(dss_filepath)

            print("Adding binary complements...")
            dss.add_binary_complements(mode="extend", skip_existing=skip_existing)

            print("Serializing updated DSS to file...")
            dss.serialize(filepath=dss_filepath)

        # Compute property with CE model
        case "compute_property_with_ce_model" | "compute_property_ce":
            model = Model(filepath=model_filepath) if model_filepath is not None else None
            _do_compute_properties(
                cem=model,
                property_name=property_name,
                property_names=property_names,
                dss_filepath=dss_filepath,
                per_formula_unit=per_formula_unit,
                linear_reference=linear_reference,
                store_ensemble_predictions=store_ensemble_predictions,
            )

        # Compute property using custom property solver
        case "compute_property_with_custom_solver":
            solver_callable, solver_kwargs = _load_property_solver(property_solver)

            _do_compute_properties(
                property_solver=solver_callable,
                property_solver_kwargs=solver_kwargs,
                property_name=property_name,
                property_names=property_names,
                dss_filepath=dss_filepath,
                per_formula_unit=per_formula_unit,
                linear_reference=linear_reference,
            )

        # Plot properties of derivative structures
        case "plot_property_vs_concentration" | "plot_property":
            dsgen = DSGenerator.from_file(dss_filepath)
            dsgen.add_fractional_concentration_binary()
            dsgen.serialize(filepath=dss_filepath)

            plat = ParentLattice(filepath=plat_filepath)
            stypes_plat = plat.get_sublattice_types()
            # Identify the binary sublattice
            binary_sublattice_id = 0
            for i, (k, v) in enumerate(stypes_plat.items()):
                if len(v) == 2:  # Binary sublattice
                    binary_sublattice_id = i
                    break

            if sset_filepath is not None:
                sset = StructuresSet(filepath=sset_filepath)
                if sset_gss_filepath is not None:
                    sset_gss = StructuresSet(filepath=sset_gss_filepath)
                else:
                    sset_gss = None
                model = Model(filepath=model_filepath) if model_filepath is not None else None
                _do_plot_properties(
                    dss_filepath,
                    property_name=property_name,
                    sset=sset,
                    sset_gss=sset_gss,
                    cem=model,
                    show_plot=show_plot,
                    save_plot=save_plot,
                    fname=plot_fname,
                    site_type=binary_sublattice_id,
                    y_axis_label=y_axis_label,
                    ymin=ymin,
                    ymax=ymax,
                )
            else:
                x = dsgen.configurations["frconc_binary"].to_numpy()

                if property_name is not None and property_names is not None:
                    raise ValueError("Only one of 'property_name' or 'property_names' should be provided, not both.")

                if property_name is not None:
                    property_names = [property_name]

                ys = []
                for property_name in property_names:
                    ys.append(dsgen.configurations[property_name].to_numpy())

                _plot_multiple_scatter(x, ys, colors, markers, sizes, save_filepath=plot_fname)

        case "plot_predictions_vs_target":
            assert sset_filepath is not None, "sset_filepath needs to be provided"
            assert model_filepath is not None, "model_filepath needs to be provided"
            assert property_name is not None, "property_name needs to be provided"
            print(
                (
                    "Plotting predictions vs. target. This is only a rough plot, "
                    "for full plot, see clusterx.visualization module"
                )
            )
            sset = StructuresSet.from_file(sset_filepath)
            cemodel = Model(filepath=model_filepath)
            plot_predictions_vs_target(
                sset=sset,
                cemodel=cemodel,
                prop_name=property_name,
                scale=1.0,
                xaxis_label=f"Calculated {property_name} [arb. units]",
                yaxis_label=f"Predicted {property_name} [arb. units]",
                show_plot=show_plot,
                save_plot=save_plot,
                fname=plot_fname,
            )
        case "plot_property_vs_concentration2":
            _do_plot_properties2(dss_filepath, mask_name=mask_name, property_names=[property_name])
        case "plot_property_vs_concentration3":
            _do_plot_properties3(dss_filepath, mask_name=mask_name, property_names=[property_name])
        case "mark_lowest_property_per_concentration":
            _do_mark_lowest(property_name, mask_name, dss_filepath)

        case "mark_lowest_and_random_properties_per_concentration":
            _do_mark_lowest_and_random(property_name, mask_name, dss_filepath, n_lowest, n_random)

        case "to_sset" | "convert_to_sset":
            # Requires
            # task = "convert_to_sset"
            # dss_filepath
            # mask_name (optional)
            # sset_filepath

            dss = DSGenerator.from_file(dss_filepath)
            sset = dss.to_structures_set(mask_name=mask_name)
            print("serializing sset")
            sset.serialize(filepath=sset_filepath, overwrite=True)

        case "merge_dsses":
            # Backward-compatible behavior:
            # - merge_mask_names (preferred): list[str|None], one per source
            # - mask_name as str: same mask used for all sources
            # - mask_name as list: treated as merge_mask_names
            masks_for_merge = merge_mask_names
            if masks_for_merge is None and isinstance(mask_name, list):
                masks_for_merge = mask_name
            elif masks_for_merge is None:
                masks_for_merge = mask_name

            merged = _do_merge_dss(
                paths=dss_filepaths,
                mask_names=masks_for_merge,
                provenance_col="source_path",
                make_ids_unique=True,
                store_source_map=True,
                check_parent_lattice=True,
            )
            merged.serialize(filepath=merged_dss_filepath)

        case 100:
            model = Model(filepath=model_filepath) if model_filepath is not None else None
            plat = ParentLattice(filepath=plat_filepath)
            sset = StructuresSet(filepath=sset_filepath)

            _do_full_enumeration(plat, sc_shape=sc_shape)
            _do_compute_properties(model)
            _do_plot_properties(sset, model)


def _do_mark_lowest(property_name, mask_name, dss_filepath):
    """Group first by fractional concentration, then mark the configuration with lowest property value in every group.
    This function is intended to be used with binary materials only.

    Creates a mask in the DSS object.
    """
    with open(dss_filepath, "rb") as f:
        dss = pickle.load(f)

    dss.add_fractional_concentration_binary()
    df = dss.configurations

    # Group by 'frconc_binary' and sort each group by 'property'
    grouped = df.groupby("frconc_binary", group_keys=False)
    sorted_groups = grouped.apply(lambda x: x.sort_values(property_name))

    # Extract the 'config_id' of the row with the minimum 'property' in each group
    marked_ids = sorted_groups.groupby("frconc_binary").first()["config_id"]

    dss.masks[mask_name] = marked_ids.to_numpy()

    with open(dss_filepath, "wb") as f:
        pickle.dump(dss, f)


def _do_mark_lowest_and_random(property_name, mask_name, dss_filepath, n_lowest, n_random):
    """
    Group by fractional concentration;
    then mark the n_lowest configurations with lowest properties per concentration and
    n_random configurations per concentration.
    This function is intended to be used with binary materials only

    Creates a mask in the DSS object
    """
    import pandas as pd

    with open(dss_filepath, "rb") as f:
        dss = pickle.load(f)

    dss.add_fractional_concentration_binary()
    df = dss.configurations

    # Group by 'fronc_binary' and sort each group by 'property'
    grouped = df.groupby("frconc_binary", group_keys=False)
    sorted_groups = grouped.apply(lambda x: x.sort_values(property_name))

    # select the n_lowest configurations with lowest property values per concentration
    marked_ids = sorted_groups.groupby("frconc_binary").head(n_lowest)["config_id"]
    # sample selects n_random random configurations; If for a specific concentration there is less than
    # n_random configurations the number of available configurations is chosen for n
    marked_ids2 = sorted_groups.groupby("frconc_binary").apply(lambda x: x.sample(n=min(n_random, len(x))))["config_id"]
    # concatenate the marked ids and exclude duplicates
    combined_ids = pd.concat([marked_ids, marked_ids2]).drop_duplicates()

    dss.masks[mask_name] = combined_ids.to_numpy()

    with open(dss_filepath, "wb") as f:
        pickle.dump(dss, f)


def _do_full_enumeration(
    plat: ParentLattice,
    nsubs_list: Optional[List[List[int]]],
    sc_sizes: Optional[List[int]] = None,
    shapes_nearest_orthogonal: Union[bool, List[int]] = False,
    sc_shape: Optional[List[List[int]]] = None,
    dss_filepath: str = "dss.pickle",
    recursive=True,
) -> None:
    dsgen = DSGenerator(plat)

    dsgen.generate(
        num_subs_list=nsubs_list,
        supercell_sizes=sc_sizes,
        shapes_nearest_orthogonal=shapes_nearest_orthogonal,
        sc_shape=sc_shape,
        recursive=recursive,
    )

    with open(dss_filepath, "wb") as f:
        pickle.dump(dsgen, f)


def process_linear_reference(
    linear_reference: Optional[Union[List[List[float]], List[dict], str]] = None,
) -> Optional[List[List[float]]]:
    if linear_reference is None:
        return None

    if isinstance(linear_reference, str):
        return linear_reference

    # If it's a list of dicts, convert to list of [x, y]
    if isinstance(linear_reference, list) and all(isinstance(item, dict) for item in linear_reference):
        try:
            return [[item["x"], item["y"]] for item in linear_reference]
        except KeyError as e:
            raise ValueError(f"Missing expected key in one of the dictionaries: {e}")

    # If already list of lists, or string, return as is
    return linear_reference


def _load_property_solver(property_solver: dict) -> tuple[Callable[..., object], dict]:
    if not isinstance(property_solver, dict):
        raise TypeError("property_solver must be a dictionary.")

    filename = property_solver.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("property_solver must include a non-empty 'filename'.")

    has_function = "function" in property_solver
    has_class = "classname" in property_solver
    if has_function == has_class:
        raise ValueError("property_solver must specify exactly one of 'function' or 'classname'.")

    solver_kwargs = property_solver.get("kwargs", {})
    if not isinstance(solver_kwargs, dict):
        raise TypeError("property_solver['kwargs'] must be a dictionary.")

    module_path = os.path.join(os.getcwd(), filename)
    spec = importlib.util.spec_from_file_location("custom_property_solver", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load custom property solver module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if has_function:
        function_name = property_solver["function"]
        solver_callable = getattr(module, function_name, None)
        if not callable(solver_callable):
            raise TypeError(f"Custom solver function {function_name!r} is missing or not callable in {module_path}.")
    else:
        class_name = property_solver["classname"]
        solver_class = getattr(module, class_name, None)
        if not callable(solver_class):
            raise TypeError(f"Custom solver class {class_name!r} is missing or not callable in {module_path}.")

        solver_instance = solver_class()
        method_name = property_solver.get("method", "compute_property")
        solver_callable = getattr(solver_instance, method_name, None)
        if not callable(solver_callable):
            raise TypeError(f"Custom solver method {method_name!r} is missing or not callable on {class_name!r}.")

    return solver_callable, solver_kwargs


def _do_compute_properties(
    cem: Optional[Model] = None,
    calculator: Optional[Calculator] = None,
    property_solver: Optional[Callable[..., float]] = None,
    property_solver_kwargs: Optional[dict] = None,
    property_name: str = "property",
    property_names: str = None,
    dss_filepath: str = "dss.pickle",
    per_formula_unit: bool = False,
    store_ensemble_predictions: bool = False,
    linear_reference: Optional[Union[List[List[float]], List[dict], str]] = None,
) -> None:
    linear_reference = process_linear_reference(linear_reference)

    with open(dss_filepath, "rb") as f:
        dss = pickle.load(f)

    if cem is not None:
        dss.compute_properties(
            property_name=property_name,
            property_names=property_names,
            cemodel=cem,
            per_formula_unit=per_formula_unit,
            linear_reference=linear_reference,
            store_ensemble_predictions=store_ensemble_predictions,
        )
    elif calculator is not None:
        dss.compute_properties(
            property_name=property_name,
            property_names=property_names,
            calculator=calculator,
            per_formula_unit=per_formula_unit,
            linear_reference=linear_reference,
        )
    elif property_solver is not None:
        dss.compute_properties(
            property_name=property_name,
            property_names=property_names,
            property_solver=property_solver,
            property_solver_kwargs=property_solver_kwargs,
            per_formula_unit=per_formula_unit,
            linear_reference=linear_reference,
        )

    with open(dss_filepath, "wb") as f:
        pickle.dump(dss, f)


def _plot_multiple_scatter(
    x: Union[List[float], np.ndarray],
    ys: List[Union[List[float], np.ndarray]],
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    sizes: Optional[List[float]] = None,
    save_filepath: Optional[str] = None,
):
    x = np.array(x)
    num_series = len(ys)

    # Set default styling if not provided
    if colors is None:
        colors = plt.cm.get_cmap("tab10").colors[:num_series]
    if markers is None:
        markers = ["o"] * num_series
    if sizes is None:
        sizes = [20] * num_series

    plt.figure(figsize=(8, 6))

    for i, y in enumerate(ys):
        y = np.array(y)
        plt.scatter(
            x,
            y,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=sizes[i % len(sizes)],
            label=f"Series {i + 1}",
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optionally save data
    if save_filepath:
        if not save_filepath.endswith(".npz"):
            save_filepath += ".npz"
        data_dict = {"x": x}
        for i, y in enumerate(ys):
            data_dict[f"y_{i}"] = np.array(y)
        np.savez(save_filepath, **data_dict)


def _do_plot_properties(
    dss_filepath,
    property_name: str = "property",
    sset=None,
    sset_gss=None,
    cem=None,
    mask_name=None,
    show_plot=True,
    save_plot=True,
    fname=None,
    site_type=0,
    y_axis_label=None,
    ymin=None,
    ymax=None,
):
    with open(dss_filepath, "rb") as f:
        dsgen = pickle.load(f)

    dsgen.add_fractional_concentration_binary()
    plot_property_vs_concentration(
        sset,
        show_loo_predictions=False,
        properties_enum=dsgen.configurations[property_name].to_numpy(),
        concentrations_enum=dsgen.configurations["frconc_binary"].to_numpy(),
        property_name=property_name,
        cemodel=cem,
        show_plot=show_plot,
        save_plot=save_plot,
        fname=fname,
        data_fname=fname + ".npz",
        site_type=site_type,
        yaxis_label=y_axis_label,
        sset_gss=sset_gss,
        ymin=ymin,
        ymax=ymax,
    )


def _do_plot_properties3(dss_filepath, mask_name=None, property_names=[]):
    import plotly.graph_objects as go

    with open(dss_filepath, "rb") as f:
        dsgen = pickle.load(f)

    dsgen.add_fractional_concentration_binary()

    configurations = dsgen.configurations
    masks = dsgen.masks

    # Separate the data for mask1
    mask1_data = configurations[configurations["config_id"].isin(masks[mask_name])]

    # Create the plot
    fig = go.Figure()

    # Add scatter points for property1
    fig.add_trace(
        go.Scatter(
            x=configurations["frconc_binary"],
            y=configurations[property_names[0]],
            mode="markers",
            marker=dict(size=8, color="blue", symbol="circle"),
            name=property_names[0],
            text=configurations["config_id"],  # Hover data
            hovertemplate="Config ID: %{text}<br>frconc_binary: %{x}<br>property1: %{y}<extra></extra>",
        )
    )

    if len(property_names) == 2:
        # Add scatter points for property2
        fig.add_trace(
            go.Scatter(
                x=configurations["frconc_binary"],
                y=configurations[property_names[1]],
                mode="markers",
                marker=dict(size=8, color="orange", symbol="triangle-up"),
                name=property_names[1],
                text=configurations["config_id"],  # Hover data
                hovertemplate="Config ID: %{text}<br>frconc_binary: %{x}<br>property2: %{y}<extra></extra>",
            )
        )

    # Add highlighted points for mask1 (property1)
    fig.add_trace(
        go.Scatter(
            x=mask1_data["frconc_binary"],
            y=mask1_data[property_names[0]],
            mode="markers",
            marker=dict(size=10, color="green", symbol="diamond"),
            name=property_names[0] + " (" + mask_name + ")",
            text=mask1_data["config_id"],  # Hover data
            hovertemplate="Config ID: %{text}<br>frconc_binary: %{x}<br>property1: %{y}<extra></extra>",
        )
    )

    # Update layout for better interactivity
    fig.update_layout(
        title="Interactive Plot with Plotly",
        xaxis_title="frconc_binary",
        yaxis_title="Properties",
        legend_title="Legend",
        template="plotly_white",
        hovermode="closest",
    )

    # Show the plot
    fig.show()


def _do_plot_properties2(dss_filepath, mask_name=None, property_names=[]):
    from bokeh.models import ColumnDataSource, CustomJSHover, HoverTool
    from bokeh.palettes import Category10
    from bokeh.plotting import figure, show

    with open(dss_filepath, "rb") as f:
        dsgen = pickle.load(f)

    dsgen.add_fractional_concentration_binary()

    configurations = dsgen.configurations.copy()
    masks = dsgen.masks

    use_jitter = True
    jitter_width = 0.003
    x_column = "frconc_binary"
    plot_x_column = x_column

    if use_jitter:
        plot_x_column = "frconc_binary_jittered"
        jitter = np.random.uniform(-jitter_width, jitter_width, size=len(configurations))
        configurations[plot_x_column] = configurations[x_column].to_numpy() + jitter

    # Prepare data sources
    source = ColumnDataSource(configurations)
    mask1_source = None
    if mask_name is not None:
        mask1_source = ColumnDataSource(configurations[configurations["config_id"].isin(masks[mask_name])])

    # Build dynamic tooltips from all DataFrame columns (skip internal jitter column).
    # Long values (e.g. ensemble-prediction arrays) are truncated by a JS formatter.
    _skip_cols = {plot_x_column} if use_jitter else set()
    _max_tooltip_chars = 80
    _truncate_fmt = CustomJSHover(
        code="""
        var s = String(value);
        if (s.length > """
        + str(80)
        + """) {
            return s.slice(0, """
        + str(80)
        + """) + '...';
        }
        return s;
    """
    )
    _tooltip_cols = [col for col in configurations.columns if col not in _skip_cols]
    # The {custom} placeholder in the format string triggers the CustomJSHover.
    dynamic_tooltips = [(col, "@{" + col + "}{custom}") for col in _tooltip_cols]
    _formatters: dict[str, str | CustomJSHover] = {"@{" + col + "}": _truncate_fmt for col in _tooltip_cols}

    # Create a Bokeh figure
    p = figure(
        title="Interactive Plot with Bokeh",
        x_axis_label="frconc_binary",
        y_axis_label=property_names[0],
        tools="pan,box_zoom,reset,save",
    )

    # Assign colors and markers for the two properties
    colors = Category10[10]
    markers = ["circle", "triangle", "square", "diamond"]

    # Plot property1 and property2 for all data
    p.scatter(
        x=plot_x_column,
        y=property_names[0],
        source=source,
        color=colors[0],
        marker=markers[0],
        legend_label=property_names[0],
        size=8,
    )
    if len(property_names) == 2:
        p.scatter(
            x=plot_x_column,
            y=property_names[1],
            source=source,
            color=colors[1],
            marker=markers[1],
            legend_label=property_names[1],
            size=8,
        )

    # Highlight property1 for mask1, if requested
    if mask_name is not None:
        assert mask1_source is not None
        p.scatter(
            x=plot_x_column,
            y=property_names[0],
            source=mask1_source,
            color=colors[2],
            marker=markers[2],
            legend_label=property_names[0] + " (" + mask_name + ")",
            size=10,
            line_width=2,
        )

    # Add hover tool to display all columns (values truncated to _max_tooltip_chars)
    hover = HoverTool(tooltips=dynamic_tooltips, formatters=_formatters)  # type: ignore[arg-type]
    p.add_tools(hover)

    # Style the plot
    if p.legend:
        p.legend[0].title = "Legend"
        p.legend[0].location = "top_left"
        p.legend[0].click_policy = "hide"  # Allow clicking to toggle visibility

    # Show the plot
    show(p)


def _do_merge_dss(
    paths: Iterable[str],
    mask_names: Optional[Union[str, Iterable[Optional[str]]]] = None,
    provenance_col: str = "source_id",
    make_ids_unique: bool = True,
    store_source_map: bool = True,
    check_parent_lattice: bool = True,
) -> "DSGenerator":
    """
    Merge multiple serialized DSGenerator objects into a new DSGenerator.

    Parameters
    ----------
    paths:
        Iterable of file paths to serialized DSGenerator objects.
    mask_names:
        Either a single mask name applied to all sources, or an iterable with one
        mask per source. A value of ``None`` for a source means include all data
        from that source.
    provenance_col:
        Column name added to `configurations` to store an integer provenance id.
    make_ids_unique:
        If True, offsets config_id and shape_id per file so they don't collide across sources.
        Recommended unless you're 100% sure ids are globally unique already.
    store_source_map:
        If True, store a mapping {source_id: path} in the returned object as `source_paths`.
    check_parent_lattice:
        If True, sanity-check that all DSS share the same parent lattice (by repr()).

    Returns
    -------
    DSGenerator
        New merged DSGenerator.
    """
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("No paths provided.")

    def _normalize_mask_token(token):
        if token is None:
            return None
        if isinstance(token, str) and token.strip().lower() in ("", "none", "null"):
            return None
        if not isinstance(token, str):
            raise ValueError(f"Each mask name must be a string or None. Got {type(token).__name__}.")
        return token

    npaths = len(paths)
    if isinstance(mask_names, str) or mask_names is None:
        masks_per_source: List[Optional[str]] = [_normalize_mask_token(mask_names)] * npaths
    else:
        masks_per_source = [_normalize_mask_token(m) for m in list(mask_names)]
        if len(masks_per_source) != npaths:
            raise ValueError(f"mask_names must have the same length as paths ({len(masks_per_source)} != {npaths}).")

    # Load first to seed the merged object
    first_path = Path(paths[0])
    dss0 = DSGenerator.from_file(first_path)

    merged = type(dss0)(dss0.plat)  # new DSGenerator with same parent lattice
    merged.property_names = set(getattr(dss0, "property_names", set()))

    # We'll build these and concat at the end (tree-reduction to avoid quadratic concat cost)
    conf_parts: List[pd.DataFrame] = []
    shape_parts: List[pd.DataFrame] = []
    merged_masks_acc: Dict[str, List[int]] = {}

    source_map: Dict[int, str] = {}

    next_config_offset = 0
    next_shape_offset = 0

    for source_id, p in enumerate(paths):
        path = Path(p)
        dss = DSGenerator.from_file(path)
        mask_name = masks_per_source[source_id]

        if check_parent_lattice:
            # Cheap-ish check. Replace with a stronger equality if you have one.
            if repr(dss.plat) != repr(dss0.plat):
                raise ValueError(f"Parent lattice mismatch for: {path}")

        merged.property_names |= set(getattr(dss, "property_names", set()))

        source_map[source_id] = str(path)

        # --- Configurations ---
        conf = dss.get_configurations(mask_name=mask_name)
        if conf is None or len(conf) == 0:
            # Still advance offsets consistently
            if make_ids_unique:
                # If empty, offsets unchanged
                pass
            continue

        conf = conf.copy()
        old_config_ids = conf["config_id"].astype("int64").to_numpy()

        # --- Shapes (only those referenced by selected configurations) ---
        shapes = dss.scell_shapes
        if shapes is not None and len(shapes) > 0:
            used_shape_ids = conf["shape_id"].astype("int64").unique()
            shapes = shapes[shapes["shape_id"].astype("int64").isin(used_shape_ids)]
            shapes = shapes.copy()

            if make_ids_unique:
                # Ensure unique shape_id across sources
                shapes["shape_id"] = shapes["shape_id"].astype("int64") + next_shape_offset

            shape_parts.append(shapes)

        # Add provenance column as small int
        conf[provenance_col] = pd.Series(source_id, index=conf.index, dtype="int32")

        if make_ids_unique:
            conf["config_id"] = conf["config_id"].astype("int64") + next_config_offset
            conf["shape_id"] = conf["shape_id"].astype("int64") + next_shape_offset

        new_config_ids = conf["config_id"].astype("int64").to_numpy()
        id_map = dict(zip(old_config_ids.tolist(), new_config_ids.tolist()))

        # Keep/remap masks from source DSS, restricted to selected configurations.
        for src_mask_name, src_mask_ids in getattr(dss, "masks", {}).items():
            src_ids = np.asarray(src_mask_ids).ravel()
            remapped_ids = [id_map[int(cid)] for cid in src_ids if int(cid) in id_map]
            if remapped_ids:
                merged_masks_acc.setdefault(str(src_mask_name), []).extend(remapped_ids)

        # Keep sigma as-is (tuple of ints => object dtype). No conversion needed.
        conf_parts.append(conf)

        # Update offsets for next file (only if uniqueness requested)
        if make_ids_unique:
            # We assume ids are non-negative integers
            next_config_offset = int(conf["config_id"].max()) + 1
            if shapes is not None and len(shapes) > 0:
                next_shape_offset = int(shapes["shape_id"].max()) + 1
            else:
                # If there were no shapes in this dss, still offset shape ids
                # based on the config's max shape_id to avoid collisions
                next_shape_offset = max(next_shape_offset, int(conf["shape_id"].max()) + 1)

    # Final concatenation (single concat each => efficient)
    if shape_parts:
        merged.scell_shapes = pd.concat(shape_parts, ignore_index=True, copy=False)
    else:
        merged.scell_shapes = pd.DataFrame(columns=["shape_id", "shape"])

    if conf_parts:
        merged.configurations = pd.concat(conf_parts, ignore_index=True, copy=False)
    else:
        merged.configurations = pd.DataFrame(columns=["config_id", "sigma", "shape_id", provenance_col])

    # Store provenance mapping
    if store_source_map:
        merged.metadata["source_paths"] = source_map
        merged.metadata["source_masks"] = {sid: masks_per_source[sid] for sid in range(len(masks_per_source))}

    merged.masks = {mname: np.asarray(mids, dtype="int64") for mname, mids in merged_masks_acc.items()}

    return merged
