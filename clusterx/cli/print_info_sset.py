# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from collections import Counter

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.cli.info_utils import report_parent_lattice, report_properties, report_shapes, start_report
from clusterx.structures_set import StructuresSet

commands = ["print_info_sset"]


@plac.pos("sset_filepath", help="Path to a StructuresSet file (.db, .json, or .xyz).")
def print_info_sset(sset_filepath: str):
    """Print structure counts, property statistics, supercell shapes, and lattice details."""
    cmd_message("head")
    report = start_report("StructuresSet information", sset_filepath)
    sset = StructuresSet.from_file(sset_filepath)
    sizes = [structure.get_natoms() for structure in sset]
    report.section("Structures")
    report.table(["Quantity", "Value"], [["Number of structures", len(sset)],
                 ["Min sites per structure", min(sizes) if sizes else 0],
                 ["Max sites per structure", max(sizes) if sizes else 0]])
    report_properties(report, sset.get_property(), len(sset))
    shapes = Counter(tuple(np.asarray(structure.get_supercell().get_transformation()).ravel()) for structure in sset)
    report_shapes(report, [(i, np.asarray(shape).reshape(3, 3), count)
                           for i, (shape, count) in enumerate(shapes.items())])
    report_parent_lattice(report, sset.get_parent_lattice())
