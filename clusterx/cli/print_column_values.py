# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.derivative_structures import DerivativeStructures

commands = ["print_column_values"]


@plac.pos(
    "dss_filepath",
    help="Path to serialized DerivativeStructures object.",
)
@plac.pos(
    "column_names",
    help="List of column names (comma-separated).",
)
@plac.opt(
    "mask_name",
    abbrev="m",
    help="Mask name used to filter the configurations before printing.",
    type=str,
)
@plac.opt(
    "show_index",
    abbrev="i",
    help="Show row index/number (default: True)",
    type=bool,
)
@plac.opt(
    "n_output_values",
    abbrev="n",
    help="Maximum number of rows to print. If omitted, all rows are printed.",
    type=int,
)
def print_column_values(
    dss_filepath: str,
    column_names: str,
    mask_name: Optional[str] = None,
    show_index: bool = True,
    n_output_values: Optional[int] = None,
):
    """Print configuration column values from a serialized DerivativeStructures."""
    cmd_message("head")

    dss = DerivativeStructures.from_file(dss_filepath)
    configurations = dss.get_configurations(mask_name=mask_name)

    if configurations is None:
        raise ValueError("No configurations available for the requested mask.")

    cols_list = [col.strip() for col in column_names.split(",") if col.strip()]
    if not cols_list:
        raise ValueError("At least one column name must be provided.")

    missing_cols = [col for col in cols_list if col not in configurations.columns]
    if missing_cols:
        raise ValueError(
            "Unknown column name(s): "
            + ", ".join(missing_cols)
            + ". Available columns: "
            + ", ".join(str(col) for col in configurations.columns)
        )

    total_rows = len(configurations)
    if n_output_values is not None:
        if n_output_values <= 0:
            raise ValueError("n_output_values must be a positive integer.")
        rows_to_print = min(total_rows, n_output_values)
    else:
        rows_to_print = total_rows
        if total_rows > 100:
            print(
                "Warning: this command will print "
                f"{total_rows} rows. "
                "Use --n-output-values (or -n) to limit output rows, "
                "e.g. -n 50."
            )
            confirmation = input("Type 'yes' to continue printing all rows: ").strip().lower()
            if confirmation != "yes":
                print("Printing cancelled.")
                return

    # Size each column to its widest displayed value, up to the existing cap.
    index_width = max(5, len(str(max(rows_to_print - 1, 0))))
    max_col_width = 30
    displayed_configurations = configurations[cols_list].head(rows_to_print)
    col_widths = {
        col: min(
            max_col_width,
            max(len(str(col)), *(len(str(value)) for value in displayed_configurations[col])),
        )
        for col in cols_list
    }

    def _crop(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    def _fmt_left(text: str, width: int) -> str:
        return _crop(text, width).ljust(width)

    def _fmt_right(text: str, width: int) -> str:
        return _crop(text, width).rjust(width)

    header = []
    if show_index:
        header.append(_fmt_right("Index", index_width))
    header.extend(_fmt_left(str(col), col_widths[col]) for col in cols_list)
    header_line = " | ".join(header)
    print(header_line)
    print("-" * len(header_line))

    for idx, (_, row_data) in enumerate(displayed_configurations.iterrows()):
        row = []
        if show_index:
            row.append(_fmt_right(str(idx), index_width))
        row.extend(_fmt_left(str(row_data[col]), col_widths[col]) for col in cols_list)
        print(" | ".join(row))

    if rows_to_print < total_rows:
        print(f"Printed {rows_to_print} of {total_rows} rows. Use --n-output-values to control this limit.")
