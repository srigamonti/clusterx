from typing import Optional, List, Callable
import pickle
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.parent_lattice import ParentLattice
from clusterx.model import Model
from clusterx.derivative_structures import DSGenerator
from clusterx.visualization import plot_property_vs_concentration
from clusterx.cli.find_lowest import find_lowest
from ase.calculators.calculator import Calculator

commands = ["generate_full_enumeration"]


def generate_full_enumeration(
    sc_sizes: Optional[List[int]] = None,
    nsubs_list: Optional[List[List[int]]] = None,
    sset_filepath: Optional[str] = None,
    sset_gss_filepath: Optional[str] = None,
    model_filepath: Optional[str] = None,
    plat_filepath: Optional[str] = None,
    dss_filepath: Optional[str] = None,
    property_label: Optional[str] = None,
    calculator_name: Optional[str] = None,
    property_solver_filename: Optional[str] = None,
    property_solver_kwargs: Optional[dict] = None,
    trafo: Optional[List[List[float]]] = None,
    per_formula_unit: bool = False,
    linear_reference: Optional[List[List[float]]] = None,
    do: int = 1,
):
    """Generate full enumeration"""

    match do:
        # Find derivative structures
        case 1:
            plat = ParentLattice(filepath=plat_filepath)
            _do_full_enumeration(
                plat, sc_sizes, nsubs_list, dss_filepath=dss_filepath, trafo=trafo
            )

        # Compute (w/CE model) and plot properties of derivative structures
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
            sset = StructuresSet(filepath=sset_filepath)
            _do_plot_properties(dss_filepath, sset=sset, cem=model)

        # Compute (w/Calculator) and plot properties of derivative structures
        case 3:
            match calculator_name:
                case "mace" | "MACE" | "Mace":
                    from mace.calculators import mace_mp

                    calculator = mace_mp(
                        model="small",
                        dispersion=False,
                        default_dtype="float64",
                        device="cpu",
                    )
            _do_compute_properties(
                calculator=calculator,
                property_label=property_label,
                dss_filepath=dss_filepath,
                per_formula_unit=per_formula_unit,
                linear_reference=linear_reference,
            )
            _do_plot_properties(dss_filepath, property_label=property_label)

        # Compute (w/custom property solver) and plot properties of derivative structures
        case 4:
            import importlib.util
            import sys
            import os

            module_path = os.path.join(os.getcwd(), property_solver_filename)
            module_name = os.path.splitext(property_solver_filename)[0]
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            _do_compute_properties(
                property_solver=module.compute_property,
                property_solver_kwargs=property_solver_kwargs,
                property_label=property_label,
                dss_filepath=dss_filepath,
                per_formula_unit=per_formula_unit,
                linear_reference=linear_reference,
            )
            _do_plot_properties(dss_filepath, property_label=property_label)

        # Plot properties of derivative structures
        case 5:
            _do_plot_properties(dss_filepath, property_label=property_label)

        case 6:
            plat = ParentLattice(filepath=plat_filepath)
            _do_find_gss(plat, sset_gss_filepath)

        case 100:
            model = (
                Model(filepath=model_filepath) if model_filepath is not None else None
            )
            plat = ParentLattice(filepath=plat_filepath)
            sset = StructuresSet(filepath=sset_filepath)

            _do_full_enumeration(plat, trafo=trafo)
            _do_compute_properties(model)
            _do_plot_properties(sset, model)


def _do_find_gss(plat, sset_gss_filepath):

    sset_gss_ = StructuresSet(plat)

    with open("dss.pickle", "rb") as f:
        dsgen = pickle.load(f)

    for i in range(len(dsgen.concentrations)):
        min_p = dsgen.properties["E_mix"][i][0]
        min_idx = 0
        for j, (p, c) in enumerate(
            zip(dsgen.properties["E_mix"][i], dsgen.concentrations[i])
        ):
            if p < min_p:
                min_p = p
                min_idx = j

        print("*************", min_idx, min_p, c)

        scell = SuperCell(parent_lattice=plat, p=dsgen.scell_shapes[i])
        struc = Structure(scell, sigmas=dsgen.sigmas[i][min_idx])
        sset_gss_.add_structure(struc, E_mix_predicted=min_p, concentration=c)

    sset_gss_.serialize("sset_temp.json", overwrite=True)

    find_lowest(
        sset_filepath="sset_temp.json",
        property_name="E_mix_predicted",
        sset_lowest_filepath=sset_gss_filepath,
    )

    # sset2 = StructuresSet(filepath=sset_gss_filepath)
    # sset_for_wahib = sset2.get_subset([9, 10, 11, 12])
    # sset_for_wahib.serialize("sset_for_wahib.json")
    # for i, s in enumerate(sset_for_wahib):
    #     write(f"geometry_{i}.in", s.get_atoms(), format="aims")


def _do_full_enumeration(
    plat: ParentLattice,
    sc_sizes: Optional[List[int]],
    nsubs_list: Optional[List[List[int]]],
    dss_filepath: str = "dss.pickle",
    trafo: List[List[int]] = None,
) -> None:

    dsgen = DSGenerator(plat)

    dsgen.generate(
        sc_sizes,
        nsubs_list,
        trafo=trafo,
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
    for i in range(len(dsgen.concentrations)):
        print(dsgen.properties[property_label][i])
        print(dsgen.concentrations[i])
        for p, c in zip(dsgen.properties[property_label][i], dsgen.concentrations[i]):
            concentrations_enum.append(c)
            properties_enum.append(p)

    plot_property_vs_concentration(
        sset,
        show_loo_predictions=False,
        properties_enum=properties_enum,
        concentrations_enum=concentrations_enum,
        property_name=property_label,
        cemodel=cem,
    )
