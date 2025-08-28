# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

from clusterx.cli.config_utils import (
    cmd_message,
    convert_to_float_list,
    convert_to_int_list,
)
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell

commands = ["build_cpool"]


@plac.pos("npoints", help="The number of points in the clusters being built.")
@plac.pos("radii", help="Corresponding cluster radii.")
@plac.opt(
    "sset_filepath",
    abbrev="ssf",
    help="Path to a serialized StructuresSet object to get the parent lattice from it.",
)
@plac.opt(
    "plat_filepath",
    abbrev="plf",
    help="Path to a serialized ParentLattice object to get the parent lattice from it.",
)
@plac.opt(
    "psc",
    abbrev="p",
    help="""
        Definition of a super cell where to find clusters, in terms of the parent lattice. 
        For a precise definition look into the parameter p of the SuperCell class of CELL.
    """,
)
@plac.opt(
    "method", abbrev="me", help="Method to use to find clusters.", choices=[1, 2, 3]
)
@plac.opt(
    "cpool_filepath", help="Filepath where to store the generated pool of clusters."
)
@plac.opt(
    "vacancy_atomic_number",
    help="Look documentation for variable vacancy_atomic_number in ClustersPool class.",
)
def build_cpool(
    npoints,  # List[int] or str
    radii,  # List[float] or str
    sset_filepath=None,  # Optional[str]
    plat_filepath=None,  # Optional[str]
    psc=1,  # Supercell definition
    method=0,  # int
    cpool_filepath="cpool.json",  # str
    vacancy_atomic_number=0,  # int
):
    """
    Build a pool of clusters.

    Parameters:
        npoints: A list of integers or a string representation of the list.
        radii: A list of floats or a string representation of the list.
        sset_filepath: Path to the structure set file (optional).
        plat_filepath: Path to the parent lattice file (optional).
        psc: Supercell definition (default: 1).
        method: Method ID used to construct clusters (default: 0).
        cpool_filepath: Output filepath for the cluster pool (default: "cpool.json").
        vacancy_atomic_number: Atomic number for vacancy site (default: 0).

    """
    if isinstance(npoints, str):
        npoints = convert_to_int_list(npoints)

    if isinstance(radii, str):
        radii = convert_to_float_list(radii)

    cmd_message("head")

    if sset_filepath is not None:
        sset = StructuresSet(filepath=sset_filepath)
        plat = sset.get_parent_lattice()

    if plat_filepath is not None:
        plat = ParentLattice(filepath=plat_filepath)
    if not (plat_filepath or sset_filepath):
        raise ValueError("Either sset_filepath or plat_filepath must be provided.")

    scell = SuperCell(plat, p=psc)
    cpool = ClustersPool(
        plat, npoints=npoints, radii=radii, super_cell=scell, method=method
    )

    cpool.print_info()
    cpool.serialize(
        filepath=cpool_filepath, vacancy_atomic_number=vacancy_atomic_number
    )
