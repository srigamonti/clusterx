# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from pathlib import Path

import plac
from ase.io import read, write

from clusterx.cli.config_utils import cmd_message

commands = ["cluster_to_cif"]


@plac.annotations(
    cpool_filepath=(
        "Serialize cluster from a ClustersPool object to cif for plotting.",
        "positional",
        None,
        str,
        None,
        None,
    ),
    cluster_index=(
        "Index of the cluster in the ClustersPool to serialize.",
        "positional",
        None,
        int,
        None,
        None,
    ),
)
def cluster_to_cif(cpool_filepath: str, cluster_index: int):
    """Serialize a cluster from a ClustersPool object to cif for plotting"""
    cmd_message("head")

    if cpool_filepath is not None:
        atoms = read(cpool_filepath, index=cluster_index)
    else:
        raise ValueError("cpool_filepath must be provided.")

    out_path = str(Path(cpool_filepath).with_suffix(""))

    write(f"{out_path}_cluster_{cluster_index}.cif", atoms, format="cif")
