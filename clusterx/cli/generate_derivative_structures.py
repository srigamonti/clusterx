# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional, List, Callable
import plac
import pickle
import importlib.util
import sys
import os
from ase.calculators.calculator import Calculator
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.model import Model
from clusterx.derivative_structures import DSGenerator
from clusterx.visualization import plot_property_vs_concentration

commands = ["generate_derivative_structures"]


@plac.opt("sc_sizes", abbrev="scsi", help="List of supercell sizes.", type=list)
@plac.opt(
    "nsubs_list",
    abbrev="nsl",
    help="""List of integer lists corresponding to every supercell 
    size to indicate the number of substitutions used in every enumeration.""",
)
@plac.opt(
    "sset_filepath",
    abbrev="ssfp",
    help="""Path to a serialized StructuresSet object. This is used in at least
    some plotting tasks""",
)
@plac.opt(
    "sset_lowest_filepath",
    abbrev="sslowest",
    help="""Path to a serialized StructuresSet object containing ground state structures. 
    This is used in at least
    some plotting tasks""",
)
@plac.opt(
    "model_filepath",
    abbrev="mfp",
    help="""Filepath of a serialized CE model object. This is used to evaluate the 
    properties enumerated configurations in at least some plotting tasks.""",
)
@plac.opt(
    "plat_filepath",
    abbrev="plfp",
    help="Filepath of a serialized ParentLattice object. This is needed to perform the enumerations.",
)
@plac.opt(
    "dss_filepath",
    abbrev="dssfp",
    help="Filepath to either serialize or retrieve an enumeration of derivative structures.",
)
@plac.opt(
    "property_label",
    abbrev="plab",
    help="Label of the property to be requested from the structures set, if present.",
)
@plac.opt(
    "property_solver",
    abbrev="psol",
    help="A dictionary to specify parameters for the property solver. See documentation above.",
    type=dict,
)
@plac.opt(
    "sc_shape",
    abbrev="scsh",
    help="3x3 matrix of integers to specify supercell shape for fixed shape enumeration.",
)
@plac.flg(
    "per_formula_unit",
    abbrev="pfu",
    help="to be changed to hanged to kwargs.",
)
@plac.opt(
    "linear_reference",
    abbrev="lref",
    help="to be changed to hanged to kwargs.",
)
@plac.opt(
    "mask_name",
    abbrev="mn",
    help="For tasks that create or use a mask, name of the mask.",
    type=str,
)
@plac.opt(
    "task",
    abbrev="task",
    help="Task to perform.",
    type=str,
)
def generate_derivative_structures(
    sc_sizes: Optional[List[int]] = None,
    nsubs_list: Optional[List[List[int]]] = None,
    sset_filepath: Optional[str] = None,
    model_filepath: Optional[str] = None,
    plat_filepath: Optional[str] = None,
    dss_filepath: Optional[str] = None,
    property_label: Optional[str] = None,
    property_solver: Optional[dict] = None,
    sc_shape: Optional[List[List[int]]] = None,
    per_formula_unit: bool = False,
    linear_reference: Optional[List[List[float]]] = None,
    mask_name: Optional[str] = None,
    task: str = "do_full_enumeration",
):
    """Generate derivative structures"""

    match task:
        # Find derivative structures
        case "do_full_enumeration":
            plat = ParentLattice(filepath=plat_filepath)

            _do_full_enumeration(
                plat,
                nsubs_list=nsubs_list,
                sc_sizes=sc_sizes,
                sc_shape=sc_shape,
                dss_filepath=dss_filepath,
            )

        # Compute property with CE model
        case "compute_property_with_ce_model":
            model = (
                Model(filepath=model_filepath) if model_filepath is not None else None
            )
            _do_compute_properties(
                cem=model,
                property_label=property_label,
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
                property_solver_kwargs=property_solver["kwargs"],
                property_label=property_label,
                dss_filepath=dss_filepath,
                per_formula_unit=per_formula_unit,
                linear_reference=linear_reference,
            )

        # Plot properties of derivative structures
        case "plot_property_vs_concentration":
            if sset_filepath is not None:
                sset = StructuresSet(filepath=sset_filepath)
                model = (
                    Model(filepath=model_filepath)
                    if model_filepath is not None
                    else None
                )
                _do_plot_properties(
                    dss_filepath, property_label=property_label, sset=sset, cem=model
                )
            else:
                _do_plot_properties(
                    dss_filepath, property_label=property_label, mask_name=mask_name
                )
        case "plot_property_vs_concentration2":
            _do_plot_properties2(
                dss_filepath, mask_name=mask_name, property_names=[property_label]
            )
        case "plot_property_vs_concentration3":
            _do_plot_properties3(
                dss_filepath, mask_name=mask_name, property_names=[property_label]
            )
        case "mark_lowest_property_per_concentration":
            _do_mark_lowest(property_label, mask_name, dss_filepath)

        case "convert_to_sset":
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

            for i in range(nstruc):
                sigma = data.iloc[i]["sigma"]
                sc_shape = sc_shapes.loc[
                    sc_shapes["shape_id"] == data.iloc[i]["shape_id"], "shape"
                ].iloc[0]
                scell = SuperCell(plat, sc_shape)
                sset.add_structure(Structure(scell, sigmas=sigma), mask=mask_name)

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


def _do_full_enumeration(
    plat: ParentLattice,
    nsubs_list: Optional[List[List[int]]],
    sc_sizes: Optional[List[int]] = None,
    sc_shape: Optional[List[List[int]]] = None,
    dss_filepath: str = "dss.pickle",
) -> None:

    dsgen = DSGenerator(plat)

    dsgen.generate(
        num_subs_list=nsubs_list,
        supercell_sizes=sc_sizes,
        sc_shape=sc_shape,
    )

    with open(dss_filepath, "wb") as f:
        pickle.dump(dsgen, f)


def _do_compute_properties(
    cem: Optional[Model] = None,
    calculator: Optional[Calculator] = None,
    property_solver: Optional[Callable[..., float]] = None,
    property_solver_kwargs: Optional[dict] = None,
    property_label: str = "property",
    dss_filepath: str = "dss.pickle",
    per_formula_unit: bool = False,
    linear_reference: Optional[List[List[float]]] = None,
) -> None:

    with open(dss_filepath, "rb") as f:
        dss = pickle.load(f)

    if cem is not None:
        dss.compute_properties(
            property_label,
            cemodel=cem,
            per_formula_unit=per_formula_unit,
            linear_reference=linear_reference,
        )
    elif calculator is not None:
        dss.compute_properties(
            property_label,
            calculator=calculator,
            per_formula_unit=per_formula_unit,
            linear_reference=linear_reference,
        )
    elif property_solver is not None:
        dss.compute_properties(
            property_label,
            property_solver=property_solver,
            property_solver_kwargs=property_solver_kwargs,
            per_formula_unit=per_formula_unit,
            linear_reference=linear_reference,
        )

    with open(dss_filepath, "wb") as f:
        pickle.dump(dss, f)


def _do_plot_properties(
    dss_filepath, property_label: str = "property", sset=None, cem=None, mask_name=None
):
    with open(dss_filepath, "rb") as f:
        dsgen = pickle.load(f)

    dsgen.add_fractional_concentration_binary()
    plot_property_vs_concentration(
        sset,
        show_loo_predictions=False,
        properties_enum=dsgen.configurations[property_label].to_numpy(),
        concentrations_enum=dsgen.configurations["frconc_binary"].to_numpy(),
        property_name=property_label,
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
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.palettes import Category10

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
