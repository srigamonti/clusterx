# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

"""Shared formatting and stored-data summaries for the print_info commands."""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import chemical_symbols

from clusterx.cli.report_utils import MarkdownReport

commands = []


def format_value(value, limit=120):
    if value is None:
        return "None"
    if isinstance(value, np.ndarray):
        value = np.array2string(value, precision=6, threshold=10, edgeitems=3)
    elif isinstance(value, (float, np.floating)):
        value = f"{value:.8g}"
    text = " ".join(str(value).split()).replace("|", "/")
    text = re.sub(r" at 0x[0-9a-fA-F]+", "", text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class InfoReport(MarkdownReport):
    """Markdown reports with tables aligned for terminal reading."""

    def table(self, headers, rows):
        rows = [[format_value(value) for value in row] for row in rows]
        headers = [str(header) for header in headers]
        widths = [max(len(header), *(len(row[i]) for row in rows), 3) for i, header in enumerate(headers)]

        def line(row):
            return "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"

        self._emit(line(headers))
        self._emit("| " + " | ".join("-" * width for width in widths) + " |")
        for row in rows:
            self._emit(line(row))
        self._emit()


def start_report(title, filepath):
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    report = InfoReport()
    report.title(title)
    report.table(["File", "Size [bytes]"], [[path, path.stat().st_size]])
    return report


def load_pickle(filepath, expected_type):
    with open(filepath, "rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected {expected_type.__name__}, got {type(obj).__name__} in '{filepath}'.")
    return obj


def as_array(values):
    try:
        return np.asarray(values)
    except ValueError:
        return np.asarray(values, dtype=object)


def numeric_stats(values):
    array = as_array(values)
    if array.size == 0 or array.dtype.kind not in "biuf":
        return ["-", "-", "-"]
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return ["-", "-", "-"]
    return [format_value(finite.min()), format_value(finite.max()), format_value(finite.mean())]


def report_properties(report, properties, sample_count, unloaded=()):
    report.section("Properties")
    count = len(properties) + len(unloaded)
    report.text(f"Number of properties: {count}. Ranges and means use finite values in the stored units.")
    rows = []
    for name, values in sorted(properties.items()):
        array = as_array(values)
        samples = list(array) if array.ndim else [array.item()]
        missing = [value is None or (np.isscalar(value) and bool(pd.isna(value))) for value in samples]
        valid = [value for value, absent in zip(samples, missing) if not absent]
        status = "yes" if len(samples) == sample_count else f"NO (expected {sample_count})"
        rows.append([name, array.shape, array.dtype, sum(missing), status, *numeric_stats(valid)])
    rows.extend([name, "not loaded", "object", "-", "-", "-", "-", "-"] for name in unloaded)
    if rows:
        report.table(
            ["Property", "Shape", "Dtype", "Missing samples", "Sample count matches", "Min", "Max", "Mean"], rows
        )
    else:
        report.text("None.")


def report_parent_lattice(report, plat):
    report.section("Parent lattice")
    types = plat.get_sublattice_types()
    counts = plat.get_nsites_per_type()
    cell = np.asarray(plat.get_cell())
    report.table(
        ["Quantity", "Value"],
        [
            ["Sites", plat.get_natoms()],
            ["Sublattices", len(types)],
            ["Substitutional sites", len(plat.get_substitutional_sites())],
            ["Spectator sites", len(plat.get_spectator_sites())],
            ["Periodic directions (a, b, c)", plat.get_pbc()],
            ["Cell volume [angstrom^3]", abs(np.linalg.det(cell))],
        ],
    )
    report.table(["Cell vector [angstrom]", "x", "y", "z"], [[axis, *row] for axis, row in zip("abc", cell)])
    report.table(
        ["Sublattice", "Sites", "Supported species", "Allowed species (in sigma order)", "Atomic numbers", "Role"],
        [
            [tag, counts[tag], len(numbers), ", ".join(chemical_symbols[int(n)] for n in numbers), numbers,
             "substitutional" if len(numbers) > 1 else "spectator"]
            for tag, numbers in sorted(types.items())
        ],
    )


def report_cluster_pool(report, pool):
    report.section("Cluster pool")
    report.text(f"Number of clusters: {len(pool)}")
    orders = np.asarray(pool.get_all_npoints())
    radii = np.asarray(pool.get_all_radii())
    multiplicities = np.asarray(pool.get_multiplicities())
    rows = []
    for order in np.unique(orders):
        selected = orders == order
        rows.append([order, int(selected.sum()), *numeric_stats(radii[selected])[:2],
                     *numeric_stats(multiplicities[selected])[:2]])
    if rows:
        report.table(["Points", "Clusters", "Min radius [angstrom]", "Max radius [angstrom]",
                      "Min multiplicity", "Max multiplicity"], rows)


def report_ccalc(report, ccalc):
    report.section("Correlation calculator")
    report.table(
        ["Quantity", "Value"],
        [
            ["Basis", ccalc.get_basis()],
            ["Correlation features / clusters", len(ccalc.get_cpool())],
            ["Symmetry-table mode", getattr(ccalc, "use_sym_table", None)],
            ["Cached supercells", len(getattr(ccalc, "_scells", []))],
            ["Cached orbit sets", len(getattr(ccalc, "_cluster_orbits_set", []))],
            ["Basis lookup shape", getattr(getattr(ccalc, "basis_set_values", None), "shape", None)],
        ],
    )
    report_cluster_pool(report, ccalc.get_cpool())
    report_parent_lattice(report, ccalc.get_cpool().get_plat())


def report_shapes(report, shapes):
    """Summarize (shape id, transformation, number of configurations) rows."""
    report.section("Supercell shapes")
    report.text(f"Number of supercell shapes: {len(shapes)}")
    if shapes:
        report.table(
            ["Shape", "Parent cells", "Configurations", "Transformation (rows)"],
            [[shape_id, int(abs(round(np.linalg.det(shape)))), count,
              "; ".join(" ".join(str(int(value)) for value in row) for row in shape)]
             for shape_id, shape, count in shapes],
        )
