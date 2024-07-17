# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac
from clusterx.cli.config_utils import get_command_name

commands = ['plot_property']

@plac.pos('filepath', help="Path to serialized structures set file")
@plac.opt('output', help="How to output the plot. 1: show on screen, 2: save to file, 3: both 1 and 2.", choices=[1,2,3])
@plac.opt('my_opt_par', abbrev='op', help="To check underscore", choices=[3,4])
@plac.flg('mark_min', abbrev='mm',help="Mark data points with lowest property value. This is useful, for instance, to indicate ground-state structures if the property is the energy.")
def plot_property(filepath, output=1, my_opt_par=3, mark_min=False):
    """Make plot of property versus concentration
    """
    command_name = get_command_name()
    config = {k: v for k, v in locals().items() if k != 'command_name'}

    print(f"Running {command_name} with configuration:")
    print(config)

