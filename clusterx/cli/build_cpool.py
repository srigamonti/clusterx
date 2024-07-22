# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import List, Tuple, Union, Optional
import inspect
import plac
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.cli.config_utils import convert_to_float_list, convert_to_int_list

Int2DArray = List[List[int]]
Int3Vector = Tuple[int, int, int]
Int2x2Matrix = List[List[int]]
Int3x3Matrix = List[List[int]]
# Define the alias for the psc type
PscType = Union[int, Int2DArray, Int3Vector, Int2x2Matrix, Int3x3Matrix]

commands = ["build_cpool"]


@plac.pos("npoints", help="The number of points in the clusters being built.")
@plac.pos("radii", help="Corresponting atom radii.")
@plac.opt(
    "sset_filepath",
    help="You can indicate a serialized structures set json file to get the parent lattice from it.",
)
@plac.opt(
    "psc",
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
def build_cpool(
    npoints: Union[List[int], str],
    radii: Union[List[float], str],
    sset_filepath: Optional[str] = None,
    psc: PscType = 1,
    method: int = 1,
    cpool_filepath: str = "cpool.json",
    vacancy_atomic_number: int = 0,
):
    """Build a pool of clusters"""
    if isinstance(npoints, str):
        npoints = convert_to_int_list(npoints)

    if isinstance(radii, str):
        radii = convert_to_float_list(radii)

    command_name = inspect.stack()[0].function
    config = {k: v for k, v in locals().items() if k != "command_name"}

    print(f"Running {command_name} with configuration:")
    print(config)

    if sset_filepath is not None:
        sset = StructuresSet(filepath=sset_filepath)
        plat = sset.get_parent_lattice()

    scell = SuperCell(plat, p=psc)
    cpool = ClustersPool(
        plat, npoints=npoints, radii=radii, super_cell=scell, method=method
    )
    cpool.display_info()
    cpool.serialize(
        filepath=cpool_filepath, vacancy_atomic_number=vacancy_atomic_number
    )
