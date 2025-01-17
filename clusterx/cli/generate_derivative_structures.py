# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional, List, Callable
import plac
import pickle
import importlib.util
import sys
import os
import numpy as np
from ase.calculators.calculator import Calculator
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.parent_lattice import ParentLattice
from clusterx.model import Model
from clusterx.derivative_structures import DSGenerator
from clusterx.visualization import plot_property_vs_concentration
from clusterx.cli.find_lowest import find_lowest

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
    "sset_gss_filepath",
    abbrev="ssgss",
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
    "task",
    abbrev="task",
    help="Task to perform.",
    type=int,
)
def generate_derivative_structures(
    sc_sizes: Optional[List[int]] = None,
    nsubs_list: Optional[List[List[int]]] = None,
    sset_filepath: Optional[str] = None,
    sset_gss_filepath: Optional[str] = None,
    model_filepath: Optional[str] = None,
    plat_filepath: Optional[str] = None,
    dss_filepath: Optional[str] = None,
    property_label: Optional[str] = None,
    property_solver: Optional[dict] = None,
    sc_shape: Optional[List[List[float]]] = None,
    per_formula_unit: bool = False,
    linear_reference: Optional[List[List[float]]] = None,
    task: int = 1,
):
    """Generate derivative structures"""

    match task:
        # Find derivative structures
        case 1:
            plat = ParentLattice(filepath=plat_filepath)
            _do_full_enumeration(
                plat, sc_sizes, nsubs_list, dss_filepath=dss_filepath, sc_shape=sc_shape
            )

        # Compute property with CE model
        case 2:
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
        case 3:

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
        case 4:
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
                _do_plot_properties(dss_filepath, property_label=property_label)

        case 5:
            plat = ParentLattice(filepath=plat_filepath)
            _do_find_gss(plat, sset_gss_filepath)

        case 100:
            model = (
                Model(filepath=model_filepath) if model_filepath is not None else None
            )
            plat = ParentLattice(filepath=plat_filepath)
            sset = StructuresSet(filepath=sset_filepath)

            _do_full_enumeration(plat, sc_shape=sc_shape)
            _do_compute_properties(model)
            _do_plot_properties(sset, model)


def _do_find_gss(plat, sset_gss_filepath):

    sset_gss_ = StructuresSet(plat)

    with open("dss.pickle", "rb") as f:
        dsgen = pickle.load(f)

    for i, concentration_set in enumerate(dsgen.concentrations):
        min_p = dsgen.properties["E_mix"][i][0]
        min_idx = 0
        min_c = concentration_set[0]
        for j, (p, c) in enumerate(
            zip(dsgen.properties["E_mix"][i], concentration_set)
        ):
            if p < min_p:
                min_p = p
                min_idx = j
                min_c = c

        print("*************", min_idx, min_p, min_c)

        scell = SuperCell(parent_lattice=plat, p=dsgen.scell_shapes[i])
        struc = Structure(scell, sigmas=dsgen.sigmas[i][min_idx])
        sset_gss_.add_structure(struc, E_mix_predicted=min_p, concentration=min_c)

    sset_gss_.serialize("sset_temp.json", overwrite=True)

    find_lowest(
        sset_filepath="sset_temp.json",
        property_name="E_mix_predicted",
        sset_lowest_filepath=sset_gss_filepath,
    )


def _do_full_enumeration(
    plat: ParentLattice,
    sc_sizes: Optional[List[int]],
    nsubs_list: Optional[List[List[int]]],
    dss_filepath: str = "dss.pickle",
    sc_shape: List[List[int]] = None,
) -> None:

    dsgen = DSGenerator(plat)

    dsgen.generate(
        supercell_sizes=sc_sizes,
        num_subs_list=nsubs_list,
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
    dss_filepath, property_label: str = "property", sset=None, cem=None
):
    with open(dss_filepath, "rb") as f:
        dsgen = pickle.load(f)

    concentrations_enum = []
    properties_enum = []

    for i, concentration_set in enumerate(dsgen.concentrations):
        print(dsgen.properties[property_label][i])
        print(concentration_set)
        for p, c in zip(dsgen.properties[property_label][i], concentration_set):
            concentrations_enum.append(c)
            properties_enum.append(p)

    plot_property_vs_concentration(
        sset,
        show_loo_predictions=False,
        properties_enum=np.array(properties_enum),
        concentrations_enum=np.array(concentrations_enum),
        property_name=property_label,
        cemodel=cem,
    )
