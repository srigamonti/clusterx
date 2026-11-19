# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.cli.info_utils import load_pickle, report_ccalc, start_report
from clusterx.correlations import CorrelationsCalculator

commands = ["print_info_ccalc"]


@plac.pos("ccalc_filepath", help="Path to a pickled CorrelationsCalculator file.")
def print_info_ccalc(ccalc_filepath: str):
    """Print the basis, cluster pool, cached orbit counts, and parent lattice."""
    cmd_message("head")
    report = start_report("CorrelationsCalculator information", ccalc_filepath)
    report_ccalc(report, load_pickle(ccalc_filepath, CorrelationsCalculator))
