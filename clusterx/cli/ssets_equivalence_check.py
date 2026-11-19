# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from typing import Optional

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.structures_set import StructuresSet
from clusterx.utils import ssets_equivalence_check as _ssets_equivalence_check

commands = ["ssets_equivalence_check"]


@plac.pos(
    "sset1_filepath",
    help="Path to the first serialized StructuresSet object.",
)
@plac.pos(
    "sset2_filepath",
    help="Path to the second serialized StructuresSet object. If omitted, compare the first set with itself.",
)
def ssets_equivalence_check(
    sset1_filepath: str,
    sset2_filepath: Optional[str] = None,
):
    """Find equivalent structures within or between serialized StructuresSet files."""
    cmd_message("head")

    sset1 = StructuresSet(filepath=sset1_filepath)
    sset2 = StructuresSet(filepath=sset2_filepath) if sset2_filepath is not None else None

    equivalent = _ssets_equivalence_check(sset1, sset2=sset2)
    equivalent_non_empty = {}
    for key, value in equivalent.items():
        value_list = value.tolist()
        if sset2_filepath is None:
            should_print = len(value_list) > 1
        else:
            should_print = bool(value_list)

        if should_print:
            equivalent_non_empty[int(key)] = value_list
    print(equivalent_non_empty)
