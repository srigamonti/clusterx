# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac
from clusterx.parent_lattice import ParentLattice
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.cli.config_utils import cmd_message

commands = ["build_ccalc"]


@plac.opt(
    "basis",
    abbrev="b",
    help="""cluster basis to be used. Possible values are: indicator-binary, trigonometric, 
        polynomial, and chebyshev. For a precise definition look into the parameter 
        'basis' of the CorrelationsCalculator class of CELL.
    """,
)
@plac.opt(
    "plat_filepath",
    abbrev="plf",
    help="Path to the json file of a serialized ParentLattice object.",
)
@plac.opt(
    "cpool_filepath",
    abbrev="cpf",
    help="Path to the json file of a serialized ClustersPool object.",
)
@plac.opt(
    "ccalc_filepath",
    abbrev="ccf",
    help="The created CorrelationsCalculator object is saved to the indicated pickle file.",
)
def build_ccalc(
    basis: str = "trigonometric",
    plat_filepath: str = "plat.json",
    cpool_filepath: str = "cpool.json",
    ccalc_filepath: str = "ccalc.pickle",
):
    """Build a CorrelationsCalculator object and serialize to pickle"""
    cmd_message("head")

    plat = ParentLattice(filepath=plat_filepath)
    cpool = ClustersPool(filepath=cpool_filepath)

    ccalc = CorrelationsCalculator(
        basis=basis, parent_lattice=plat, clusters_pool=cpool
    )
    ccalc.serialize(filepath=ccalc_filepath)
