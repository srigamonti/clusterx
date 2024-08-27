from typing import Optional, List
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.parent_lattice import ParentLattice
from clusterx.model import Model
from clusterx.derivative_structures import DSGenerator
from clusterx.visualization import plot_property_vs_concentration
from clusterx.cli.find_lowest import find_lowest
import pickle

commands = ["generate_full_enumeration"]


def generate_full_enumeration(
    sset_filepath: Optional[str] = None,
    sset_gss_filepath: Optional[str] = None,
    model_filepath: Optional[str] = None,
    plat_filepath: Optional[str] = None,
    trafo: Optional[List[List[float]]] = None,
    do: int = 1,
):
    """Generate full enumeration"""

    if do == 1:
        model = Model(filepath=model_filepath) if model_filepath is not None else None
        plat = ParentLattice(filepath=plat_filepath)
        sset = StructuresSet(filepath=sset_filepath)

        _do_full_enumeration(plat, trafo=trafo)
        _do_compute_properties(model)
        _do_plot_properties(sset, model)

    elif do == 2:
        plat = ParentLattice(filepath=plat_filepath)
        _do_find_gss(plat, sset_gss_filepath)


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


def _do_full_enumeration(plat, trafo=None):

    dsgen = DSGenerator(plat)

    if 0:
        dsgen.generate(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [
                [0, 1],  # 1
                [1, 2],  # 2
                [1, 2, 3],  # 3
                [1, 2, 3, 4, 5],  # 4
                [1, 2, 3, 4, 5, 6],  # 5
                [1, 2, 3, 4, 5, 6, 7],  # 6
                [1, 2, 3, 4, 5, 6, 7, 8],  # 7
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 8
            ],
            trafo=trafo,
        )

    if 0:
        dsgen.generate(
            [1, 2, 3, 4, 5],
            [
                [0, 1],  # 1
                [1, 2],  # 2
                [1, 2, 3],  # 3
                [1, 2, 3, 4, 5],  # 4
                [1, 2, 3, 4, 5, 6],  # 5
            ],
            trafo=trafo,
        )

    if 0:
        dsgen.generate(
            [1, 2, 3, 4],
            [
                [0, 1],  # 1
                [1, 2],  # 2
                [1, 2, 3],  # 3
                [1, 2, 3, 4, 5],  # 4
            ],
            trafo=trafo,
        )

    if 1:
        dsgen.generate(
            [27],
            [
                [0, 1, 2, 3, 4],
            ],
            trafo=trafo,
        )

    with open("dss.pickle", "wb") as f:
        pickle.dump(dsgen, f)


def _do_compute_properties(cem):

    with open("dss.pickle", "rb") as f:
        dsgen = pickle.load(f)

    dsgen.compute_properties("E_mix", cem)

    with open("dss.pickle", "wb") as f:
        pickle.dump(dsgen, f)


def _do_plot_properties(sset, cem):
    with open("dss.pickle", "rb") as f:
        dsgen = pickle.load(f)

    concentrations_enum = []
    properties_enum = []
    for i in range(len(dsgen.concentrations)):
        for p, c in zip(dsgen.properties["E_mix"][i], dsgen.concentrations[i]):
            concentrations_enum.append(c)
            properties_enum.append(p)

    plot_property_vs_concentration(
        sset,
        show_loo_predictions=False,
        properties_enum=properties_enum,
        concentrations_enum=concentrations_enum,
        property_name="E_mix",
        cemodel=cem,
    )
