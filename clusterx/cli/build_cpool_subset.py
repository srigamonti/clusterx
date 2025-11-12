# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

from clusterx.cli.config_utils import (
    cmd_message,
)
from clusterx.clusters.clusters_pool import ClustersPool

commands = ["build_cpool_subset"]


@plac.pos(
    "cluster_indexes",
    help="List of cluster indices to manually select clusters out of the given clusters pool.",
)
@plac.opt(
    "cpool_filepath_read",
    abbrev="cfr",
    help="Filepath where to read the pool of clusters from which a subset will be generated.",
)
@plac.opt(
    "cpool_filepath_write",
    abbrev="cfw",
    help="Filepath where to store the generated sub-pool of clusters.",
)
def build_cpool_subset(
    cluster_indexes,  # List[int]
    cpool_filepath_read="cpool.json",  # str
    cpool_filepath_write="cpool-subset.json",  # str
):
    """
    Build a pool of clusters.

    Parameters:
        cluster_indices: list of int
        cpool_filepath_read: Filepath where to read the pool of clusters from which a subset will be generated..
        cpool_filepath_write: Filepath where to store the generated sub-pool of clusters.

    """
    cmd_message("head")

    cpool = ClustersPool(filepath=cpool_filepath_read)

    cpool_subset = cpool.get_subpool(cluster_indexes=cluster_indexes)

    print("\n" + "=" * 72)
    print("Original clusters pool")
    print("=" * 72)
    cpool.print_info()

    print("\n" + "=" * 72)
    print("Created subset of original clusters pool")
    print("=" * 72)
    cpool_subset.print_info()

    cpool_subset.serialize(filepath=cpool_filepath_write)
