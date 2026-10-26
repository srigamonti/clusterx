# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import ast
import os

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.structure import Structure
from clusterx.utils import normalize_shape_input, super_structure

commands = ["build_superstructure"]


@plac.pos(
    "structure_filepath",
    help="Path to a serialized Structure object (.json or .pickle).",
)
@plac.opt(
    "ss_shape",
    abbrev="sssh",
    help=(
        "Superstructure shape. Can be an integer, a 3-vector, or a 3x3 integer matrix. "
        "Examples: 2, [2,2,2], [[2,0,0],[0,2,0],[0,0,2]]."
    ),
)
@plac.opt(
    "superstructure_filepath",
    abbrev="ssfp",
    help=("Output filepath for the generated superstructure. By default, this is <input_name>_superstructure<ext>."),
)
def build_superstructure(
    structure_filepath: str,
    ss_shape="2",
    superstructure_filepath=None,
):
    """Build and serialize a superstructure from a serialized Structure object."""
    cmd_message("head")

    if superstructure_filepath is None:
        stem, ext = os.path.splitext(structure_filepath)
        superstructure_filepath = f"{stem}_superstructure{ext}"

    shape_obj = _parse_ss_shape(ss_shape)
    shape = normalize_shape_input(shape_obj)

    if shape is None:
        raise ValueError("ss_shape cannot be None.")

    structure = Structure.from_file(structure_filepath)
    structure_super = super_structure(structure, shape)
    structure_super.serialize(filepath=superstructure_filepath)

    print(f"Superstructure serialized to {superstructure_filepath}")


def _parse_ss_shape(ss_shape):
    """Parse CLI/TOML ss_shape input to int/list/matrix before normalization."""
    if isinstance(ss_shape, str):
        value = ss_shape.strip()
        try:
            ss_shape = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Allow plain scalar values passed without brackets, e.g. --ss-shape 2
            try:
                ss_shape = int(value)
            except ValueError as exc:
                raise ValueError("Could not parse ss_shape. Use an int, a 3-vector, or a 3x3 matrix.") from exc

    if not isinstance(ss_shape, (int, list, tuple)):
        raise ValueError("ss_shape must be an int, list, tuple, or string representation of these.")

    return ss_shape
