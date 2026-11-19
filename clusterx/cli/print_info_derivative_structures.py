# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.cli.info_utils import load_pickle, report_parent_lattice, report_properties, report_shapes, start_report
from clusterx.derivative_structures import DerivativeStructures

commands = ["print_info_derivative_structures", "print_info_dss"]


@plac.pos("dss_filepath", help="Path to a pickled DerivativeStructures file.")
def print_info_derivative_structures(dss_filepath: str):
    """Print configurations, properties, masks, supercell shapes, metadata, and lattice details."""
    cmd_message("head")
    report = start_report("DerivativeStructures summary", dss_filepath)
    dss = load_pickle(dss_filepath, DerivativeStructures)
    report.section("Configurations")
    report.text(f"Number of configurations: {len(dss.configurations)}")
    report.table(["Column", "Dtype"], [[name, dtype] for name, dtype in dss.configurations.dtypes.items()])
    report_properties(report, {name: dss.configurations[name].tolist() for name in dss.get_property_names()
                               if name in dss.configurations}, len(dss.configurations))
    report.section("Masks")
    report.text(f"Number of masks: {len(dss.masks)}")
    if dss.masks:
        report.table(["Mask", "Configuration IDs"], [[name, len(ids)] for name, ids in sorted(dss.masks.items())])
    counts = dss.configurations["shape_id"].value_counts()
    report_shapes(report, [(row.shape_id, row.shape, int(counts.get(row.shape_id, 0)))
                           for row in dss.scell_shapes.itertuples(index=False)])
    report.section("Metadata")
    if dss.metadata:
        report.table(["Key", "Value"], sorted(dss.metadata.items()))
    else:
        report.text("None stored.")
    report_parent_lattice(report, dss.plat)


print_info_dss = print_info_derivative_structures
