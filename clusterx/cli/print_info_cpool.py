# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.cli.info_utils import report_cluster_pool, report_parent_lattice, start_report
from clusterx.clusters.clusters_pool import ClustersPool

commands = ["print_info_cpool"]


@plac.pos("cpool_filepath", help="Path to a serialized ClustersPool file.")
def print_info_cpool(cpool_filepath: str):
    """Print cluster counts by order, radii, multiplicities, and lattice details."""
    cmd_message("head")
    report = start_report("ClustersPool information", cpool_filepath)
    pool = ClustersPool(filepath=cpool_filepath)
    report_cluster_pool(report, pool)
    report_parent_lattice(report, pool.get_plat())
