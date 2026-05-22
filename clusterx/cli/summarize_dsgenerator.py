# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.derivative_structures import DSGenerator

commands = ["summarize_dsgenerator"]


@plac.annotations(
    dsg_filepath=(
        "Path to serialized DSGenerator object.",
        "positional",
        None,
        str,
    ),
)
def summarize_dsgenerator(dsg_filepath: Optional[str] = None):
    """Print a summary of a serialized DSGenerator object."""
    cmd_message("head")

    if dsg_filepath is None:
        raise ValueError("dsg_filepath must be provided.")

    dsg = DSGenerator.from_file(dsg_filepath)

    print("DSGenerator summary")
    print("=" * 72)

    # configurations dataframe
    print("\nConfigurations")
    print("-" * 72)
    print(f"Number of configurations: {len(dsg.configurations)}")
    print("Columns in configurations:")
    for col in dsg.configurations.columns:
        print(f"  - {col}")

    # masks
    print("\nMasks")
    print("-" * 72)
    mask_names = sorted(dsg.masks.keys())
    print(f"Number of masks: {len(mask_names)}")
    if mask_names:
        for name in mask_names:
            try:
                nconf = len(dsg.get_mask(name))
                print(f"  - {name} ({nconf} config_ids)")
            except Exception:
                print(f"  - {name}")
    else:
        print("  None")

    # property names
    print("\nProperty names")
    print("-" * 72)
    property_names = sorted(dsg.get_property_names())
    print(f"Number of property names: {len(property_names)}")
    if property_names:
        for name in property_names:
            print(f"  - {name}")
    else:
        print("  None")

    # supercell shapes
    print("\nSupercell shapes")
    print("-" * 72)
    n_shapes = len(dsg.scell_shapes)
    print(f"Number of supercell shapes: {n_shapes}")

    if n_shapes > 0:
        cols = dsg.scell_shapes.columns.tolist()

        has_shape_id = "shape_id" in cols
        has_size = "size" in cols

        for _, row in dsg.scell_shapes.iterrows():
            if has_shape_id and has_size:
                print(f"  - shape_id={row['shape_id']}, size={row['size']}")
            elif has_shape_id:
                try:
                    size = dsg.get_scell_size(row["shape_id"])
                    print(f"  - shape_id={row['shape_id']}, size={size}")
                except Exception:
                    print(f"  - shape_id={row['shape_id']}")
            else:
                print("  - shape entry found")
    else:
        print("  None")

    print("\nDone.")
