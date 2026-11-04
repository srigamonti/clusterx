# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.cli.info_utils import report_parent_lattice, start_report
from clusterx.parent_lattice import ParentLattice

commands = ["print_info_plat"]


@plac.pos("plat_filepath", help="Path to a serialized ParentLattice file.")
def print_info_plat(plat_filepath: str):
    """Print lattice geometry, sublattices, allowed species, and periodicity."""
    cmd_message("head")
    report = start_report("ParentLattice information", plat_filepath)
    report_parent_lattice(report, ParentLattice(filepath=plat_filepath))
