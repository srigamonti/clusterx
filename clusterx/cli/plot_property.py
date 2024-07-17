# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac
from clusterx.cli.config_utils import get_command_name
from clusterx.visualization import plot_property_vs_concentration
from clusterx.structures_set import StructuresSet

commands = ['plot_property']

@plac.pos('filepath', 
          help="Path to serialized structures set file")
@plac.pos('property_name', 
          help="Name of the property to be plotted")
@plac.opt('output', 
          help="How to output the plot. 1: show on screen, 2: save to file, 3: both 1 and 2.", 
          choices=[1,2,3])
@plac.opt('fig_fname', 
          abbrev='fn', 
          help="filepath to save generated figure file if output is 2 or 3")
@plac.flg('mark_min', 
          abbrev='mm',
          help="Mark data points with lowest property value. This is useful, for instance, to indicate ground-state structures if the property is the energy.")
def plot_property(filepath, property_name, output=1, fig_fname=None, mark_min=False):
    """Make plot of property versus concentration
    """
    command_name = get_command_name()
    config = {k: v for k, v in locals().items() if k != 'command_name'}

    print(f"Running {command_name} with configuration:")
    print(config)

    show_plot = True
    if output == 2:
        show_plot = False

    if  output != 1:
        if fig_fname is None:
            fig_fname = "plot_property.png"    

    sset = StructuresSet(filepath=filepath)

    plot_property_vs_concentration(
        sset, 
        property_name,
        show_plot=show_plot, 
        fig_fname=fig_fname)
