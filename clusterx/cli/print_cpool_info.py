# Copyright (c) 2015-2025, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.clusters.clusters_pool import ClustersPool

commands = ["print_cpool_info"]


@plac.annotations(
    cpool_filepath=(
        "Path to serialized ClustersPool object.",
        "positional",
        None,
        str,
        None,
        None,
    ),
)
def print_cpool_info(cpool_filepath: str):
    """Print ClustersPool info"""
    cmd_message("head")

    if cpool_filepath is not None:
        cpool = ClustersPool(filepath=cpool_filepath)
    else:
        raise ValueError("cpool_filepath must be provided.")

    cpool.print_info()
