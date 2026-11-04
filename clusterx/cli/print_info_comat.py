# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.cli.info_utils import numeric_stats, report_properties, start_report

commands = ["print_info_comat"]


@plac.pos("comat_filepath", help="Path to a correlation-matrix NPZ file.")
def print_info_comat(comat_filepath: str):
    """Print correlation dimensions, numeric quality, properties, and stored arrays."""
    cmd_message("head")
    report = start_report("Correlation matrix information", comat_filepath)
    with np.load(comat_filepath, allow_pickle=False) as archive:
        if "comat" not in archive.files:
            raise ValueError("Correlation-matrix archive must contain a 'comat' array.")
        matrix = archive["comat"]
        if matrix.ndim != 2 or matrix.dtype.kind not in "biufc":
            raise ValueError("The 'comat' array must be a two-dimensional numeric matrix.")
        report.section("Correlation matrix")
        report.table(
            ["Quantity", "Value"],
            [["Samples", matrix.shape[0]], ["Features", matrix.shape[1]],
             ["Shape", matrix.shape], ["Dtype", matrix.dtype], ["Memory [bytes]", matrix.nbytes],
             ["Finite values", int(np.isfinite(matrix).sum())],
             ["NaN values", int(np.isnan(matrix).sum())],
             ["Infinite values", int(np.isinf(matrix).sum())]],
        )
        report.table(["Min", "Max", "Mean"], [numeric_stats(matrix)])
        arrays = []
        properties = {}
        unloaded = []
        for name in archive.files:
            try:
                array = matrix if name == "comat" else archive[name]
            except ValueError as exc:
                if "Object arrays cannot be loaded" not in str(exc):
                    raise
                arrays.append([name, "not loaded", "object"])
                if name.startswith("property_"):
                    unloaded.append(name[len("property_"):])
                continue
            arrays.append([name, array.shape, array.dtype])
            if name.startswith("property_"):
                properties[name[len("property_"):]] = array
        report_properties(report, properties, matrix.shape[0], unloaded=unloaded)
        report.section("Stored arrays")
        report.table(["Array", "Shape", "Dtype"], arrays)
