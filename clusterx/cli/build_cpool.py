# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import List, Tuple, Union, Optional
import plac
from clusterx.structures_set import StructuresSet
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.cli.config_utils import convert_to_float_list, convert_to_int_list
from clusterx.cli.config_utils import cmd_message

Int2DArray = List[List[int]]
Int3Vector = Tuple[int, int, int]
Int3Array = List[int]
Int2x2Matrix = List[List[int]]
Int3x3Matrix = List[List[int]]
# Define the alias for the psc type
PscType = Union[str, int, Int2DArray, Int3Vector, Int3Array, Int2x2Matrix, Int3x3Matrix]

commands = ["build_cpool"]


@plac.pos("npoints", help="The number of points in the clusters being built.")
@plac.pos("radii", help="Corresponting cluster radii.")
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
    help=(
        "Definition of a super cell where to find clusters, in terms of the parent lattice. "
        "For a precise definition look into the parameter p of the SuperCell class of CELL."
    ),
)
@plac.opt("method", help="Method to use to find clusters.", choices=[1, 2, 3])
@plac.opt(
    "cpool_filepath", help="Filepath where to store the generated pool of clusters."
)
@plac.opt(
    "vacancy_atomic_number",
    help="Look documentation for variable vacancy_atomic_number in ClustersPool class.",
)
@plac.opt(
    "mlims",
    help=(
        "Only clusters with multiplicity m larger or equal to nlims[0] and smaller or "
        "equal to nlims[1] are included in the built ClustersPool.",
    ),
)
def build_cpool(
    npoints: Union[List[int], str],
    radii: Union[List[float], str],
    sset_filepath: Optional[str] = None,
    plat_filepath: Optional[str] = None,
    psc: PscType = 1,
    method: int = 1,
    cpool_filepath: str = "cpool.json",
    vacancy_atomic_number: int = 0,
    mlims: Optional[List[int]] = None,
):
    """Build a pool of clusters"""
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

    scell = SuperCell(plat, p=psc)
    cpool = ClustersPool(
        plat, npoints=npoints, radii=radii, super_cell=scell, method=method
    )

    if mlims is not None:
        multiplicities = cpool.get_multiplicities()
        cidxs = []
        for ic, m in enumerate(multiplicities):
            if m >= mlims[0] and m <= mlims[1]:
                cidxs.append(ic)

        cpool = cpool.get_subpool(cidxs)

    cpool.print_info()
    cpool.serialize(
        filepath=cpool_filepath, vacancy_atomic_number=vacancy_atomic_number
    )
