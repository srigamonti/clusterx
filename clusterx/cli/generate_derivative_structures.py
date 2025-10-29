# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import importlib.util
import os
import pickle
import sys
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import plac
from ase.calculators.calculator import Calculator

from clusterx.derivative_structures import DSGenerator
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
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
    n_random=("Number of random structures to include.", "option", None, int),
    random_state=("Seed for random number generators.", "option", None, int),
    task=("Task to perform.", "option", "task", str),
    non_recursive=(
        "Use slow non-recursive method for finding derivative structures",
        "flag",
        "rec",
        bool,
    ),
)
def generate_derivative_structures(
    sc_sizes: Optional[List[int]] = None,
    shapes_nearest_orthogonal: Union[bool, List[int]] = False,
    nsubs_list: Optional[Union[int, List[int], List[List[int]]]] = None,
    sset_filepath: Optional[str] = None,
    model_filepath: Optional[str] = None,
    plat_filepath: Optional[str] = None,
    dss_filepath: Optional[str] = "dss.pickle",
    property_name: Optional[str] = None,
    property_names: Optional[List[str]] = None,
    property_solver: Optional[dict] = None,
    sc_shape: Optional[Union[int, List[int], List[List[int]]]] = None,
    per_formula_unit: bool = False,
    linear_reference: Optional[Union[List[List[float]], List[dict], str]] = None,
    mask_name: Optional[str] = None,
    n_lowest: Optional[int] = 1,
    n_random: Optional[int] = 0,
    random_state: Optional[int] = None,
    task: str = "do_full_enumeration",
    plotdata_filepath=None,
    colors=None,
    markers=None,
    sizes=None,
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
    property_name : str, optional
        Single property name to compute/plot. Mutually exclusive with ``property_names``.
    property_names : list[str], optional
        Multiple property names to compute/plot. Mutually exclusive with ``property_name``.
    property_solver : dict, optional
        Parameters for a custom property solver. Expected keys:
        ``filename`` (module path), ``classname`` (class with ``compute_property``),
        and optional ``kwargs`` forwarded to the solver.
    sc_shape : int or list[int] or list[list[int]], optional
        Supercell shape specification. May be a scalar multiplier or an explicit 3×3 integer matrix.
    per_formula_unit : bool, default ``False``
        If ``True``, compute/scale properties per formula unit.
    linear_reference : list[list[float]] or list[dict] or str, optional
        Linear reference to correct the property. Accepts ``[[x, y], ...]`` or
        ``[{'x': x, 'y': y}, ...]``; strings are passed through unchanged.
    mask_name : str, optional
        Name of a mask to write/read within the serialized DSS object for downstream filtering.
    n_lowest : int, default ``1``
        Count of lowest-value configurations per concentration to mark (used by
        ``"mark_lowest_property_per_concentration"`` / ``"mark_lowest_and_random_properties_per_concentration"``).
    n_random : int, default ``0``
        Number of random configurations per concentration to also mark (used by the combined marking task).
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
        Randomly generate unique configurations under the given size/shape/substitution constraints.
        Writes the resulting DSS to ``dss_filepath``.
    ``"do_full_enumeration"`` or ``"find_derivative_structures"``
        Exhaustively enumerate derivative structures for the given lattice/sizes/shapes.
        Writes the resulting DSS to ``dss_filepath``. Honor ``non_recursive``.
    ``"compute_property_with_ce_model"`` or ``"compute_property_ce"``
        Compute ``property_name`` (or ``property_names``) for all configurations in ``dss_filepath``
        using the CE model from ``model_filepath``. Supports ``per_formula_unit`` and ``linear_reference``.
    ``"compute_property_with_custom_solver"``
        Load the solver class from ``property_solver['filename']`` and call its
        ``compute_property(structure, shape, concentration, **kwargs)`` on every configuration.
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
        Create a mask with the ``n_lowest`` best plus ``n_random`` random configurations per concentration.
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

    **2) Compute a property with a custom solver and store values inside the DSS**

    .. code-block:: toml

        [generate_derivative_structures]
        task = "compute_property_with_custom_solver"
        plat_filepath = "plat.json"
        property_solver = { filename = "path/to/my_custom_solver_module.py",
                            classname = "MySolver",
                            kwargs = { arg1 = 5.724589, arg2 = 6.016160 } }
        dss_filepath = "dss_scsize1-2_new.pickle"
        property_name = "total_energy_mace_vegards"
        per_formula_unit = true
        linear_reference = [{x=0.0, y=-13.08516}, {x=1.0, y=-11.302472}]

    **Custom solver interface**

    .. code-block:: python

        class MySolver:
            def __init__(self):
                pass

            def compute_property(self, struc, shape, conc, **kwargs) -> float:
                # return a scalar property value
                return 0.0
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

        # Compute property with CE model
        case "compute_property_with_ce_model" | "compute_property_ce":
            model = (
                Model(filepath=model_filepath) if model_filepath is not None else None
            )
            _do_compute_properties(
                cem=model,
                property_name=property_name,
                dss_filepath=dss_filepath,
                per_formula_unit=per_formula_unit,
                linear_reference=linear_reference,
            )

        # Compute property using custom property solver
        case "compute_property_with_custom_solver":
            module_path = os.path.join(os.getcwd(), property_solver["filename"])
            spec = importlib.util.spec_from_file_location(
                "custom_property_solver", module_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["custom_property_solver"] = module
            spec.loader.exec_module(module)

            property_solver_class = getattr(module, property_solver["classname"])

            property_solver_instance = property_solver_class()

            _do_compute_properties(
                property_solver=property_solver_instance.compute_property,
                property_solver_kwargs=property_solver.get("kwargs", {}),
                property_name=property_name,
                property_names=property_names,
                dss_filepath=dss_filepath,
                per_formula_unit=per_formula_unit,
                linear_reference=linear_reference,
            )

        # Plot properties of derivative structures
        case "plot_property_vs_concentration" | "plot_property":
            if sset_filepath is not None:
                sset = StructuresSet(filepath=sset_filepath)
                model = (
                    Model(filepath=model_filepath)
                    if model_filepath is not None
                    else None
                )
                _do_plot_properties(
                    dss_filepath, property_name=property_name, sset=sset, cem=model
                )
            else:
                # _do_plot_properties(dss_filepath, property_name=property_name, mask_name=mask_name)
                with open(dss_filepath, "rb") as f:
                    dsgen = pickle.load(f)

                dsgen.add_fractional_concentration_binary()
                x = dsgen.configurations["frconc_binary"].to_numpy()

                if property_name is not None and property_names is not None:
                    raise ValueError(
                        "Only one of 'property_name' or 'property_names' should be provided, not both."
                    )

                if property_name is not None:
                    property_names = [property_name]

                ys = []
                for property_name in property_names:
                    ys.append(dsgen.configurations[property_name].to_numpy())

                _plot_multiple_scatter(
                    x, ys, colors, markers, sizes, save_filepath=plotdata_filepath
                )

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
            sset = StructuresSet(filepath=sset_filepath)
            cemodel = Model(filepath=model_filepath)
            plot_predictions_vs_target(
                sset=sset,
                cemodel=cemodel,
                prop_name=property_name,
                scale=1.0,
                xaxis_label=f"Calculated {property_name} [arb. units]",
                yaxis_label=f"Predicted {property_name} [arb. units]",
            )
        case "plot_property_vs_concentration2":
            _do_plot_properties2(
                dss_filepath, mask_name=mask_name, property_names=[property_name]
            )
        case "plot_property_vs_concentration3":
            _do_plot_properties3(
                dss_filepath, mask_name=mask_name, property_names=[property_name]
            )
        case "mark_lowest_property_per_concentration":
            _do_mark_lowest(property_name, mask_name, dss_filepath)

        case "mark_lowest_and_random_properties_per_concentration":
            _do_mark_lowest_and_random(
                property_name, mask_name, dss_filepath, n_lowest, n_random
            )

        case "to_sset" | "convert_to_sset":
            # Requires
            # task = "convert_to_sset"
            # plat_filepath
            # dss_filepath
            # mask_name (optional)
            # sset_filepath

            plat = ParentLattice(filepath=plat_filepath)
            sset = StructuresSet(parent_lattice=plat)
            with open(dss_filepath, "rb") as f:
                dss = pickle.load(f)

            configurations = dss.configurations
            masks = dss.masks
            sc_shapes = dss.scell_shapes

            data = (
                configurations[configurations["config_id"].isin(masks[mask_name])]
                if mask_name is not None
                else configurations
            )

            nstruc = len(data)

            print("Creating sset object")
            scell_cache = {}
            for i in range(nstruc):
                sigma = data.iloc[i]["sigma"]
                shape_id = data.iloc[i]["shape_id"]

                if shape_id not in scell_cache:
                    sc_shape = sc_shapes.loc[
                        sc_shapes["shape_id"] == shape_id, "shape"
                    ].iloc[0]
                    scell_cache[shape_id] = SuperCell(plat, sc_shape)

                scell = scell_cache[shape_id]
                sset.add_structure(Structure(scell, sigmas=sigma), mask=mask_name)

            for property_name in dss.get_property_names():
                sset.set_property_values(
                    property_name=property_name,
                    property_vals=data[property_name].tolist(),
                )

            print("serializing sset")
            sset.serialize(filepath=sset_filepath, overwrite=True)

        case 100:
            model = (
                Model(filepath=model_filepath) if model_filepath is not None else None
            )
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


def _do_mark_lowest_and_random(
    property_name, mask_name, dss_filepath, n_lowest, n_random
):
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
    marked_ids2 = sorted_groups.groupby("frconc_binary").apply(
        lambda x: x.sample(n=min(n_random, len(x)))
    )["config_id"]
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
    if isinstance(linear_reference, list) and all(
        isinstance(item, dict) for item in linear_reference
    ):
        try:
            return [[item["x"], item["y"]] for item in linear_reference]
        except KeyError as e:
            raise ValueError(f"Missing expected key in one of the dictionaries: {e}")

    # If already list of lists, or string, return as is
    return linear_reference


def _do_compute_properties(
    cem: Optional[Model] = None,
    calculator: Optional[Calculator] = None,
    property_solver: Optional[Callable[..., float]] = None,
    property_solver_kwargs: Optional[dict] = None,
    property_name: str = "property",
    property_names: str = None,
    dss_filepath: str = "dss.pickle",
    per_formula_unit: bool = False,
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
    dss_filepath, property_name: str = "property", sset=None, cem=None, mask_name=None
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
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.palettes import Category10
    from bokeh.plotting import figure, show

    with open(dss_filepath, "rb") as f:
        dsgen = pickle.load(f)

    dsgen.add_fractional_concentration_binary()

    configurations = dsgen.configurations
    masks = dsgen.masks

    # Prepare data sources
    source = ColumnDataSource(configurations)
    mask1_source = ColumnDataSource(
        configurations[configurations["config_id"].isin(masks[mask_name])]
    )

    # Create a Bokeh figure
    p = figure(
        title="Interactive Plot with Bokeh",
        x_axis_label="frconc_binary",
        y_axis_label=property_names[0],
        tools="pan,box_zoom,reset,save",
        tooltips=[("Config ID", "@config_id"), ("frconc_binary", "@frconc_binary")],
    )

    # Assign colors and markers for the two properties
    colors = Category10[10]
    markers = ["circle", "triangle", "square", "diamond"]

    # Plot property1 and property2 for all data
    p.scatter(
        x="frconc_binary",
        y=property_names[0],
        source=source,
        color=colors[0],
        marker=markers[0],
        legend_label=property_names[0],
        size=8,
    )
    if len(property_names) == 2:
        p.scatter(
            x="frconc_binary",
            y=property_names[1],
            source=source,
            color=colors[1],
            marker=markers[1],
            legend_label=property_names[1],
            size=8,
        )

    # Highlight property1 for mask1
    p.scatter(
        x="frconc_binary",
        y=property_names[0],
        source=mask1_source,
        color=colors[2],
        marker=markers[2],
        legend_label=property_names[0] + " (" + mask_name + ")",
        size=10,
        line_width=2,
    )

    # Add hover tool to display config_id
    hover = HoverTool(
        tooltips=[("Config ID", "@config_id"), ("frconc_binary", "@frconc_binary")]
    )
    p.add_tools(hover)

    # Style the plot
    p.legend.title = "Legend"
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"  # Allow clicking to toggle visibility

    # Show the plot
    show(p)
