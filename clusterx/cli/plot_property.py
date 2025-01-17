# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional
import plac
from clusterx.cli.config_utils import cmd_message
from clusterx.visualization import plot_property_vs_concentration
from clusterx.structures_set import StructuresSet

commands = ["plot_property"]


@plac.pos("sset_filepath", help="Path to file of serialized StructuresSet object")
@plac.pos("property_name", help="Name of the property to be plotted")
@plac.opt(
    "sset_gss_filepath",
    help="Path to file of serialized StructuresSet object for GS structures",
)
@plac.opt("model_filepath", help="Path to file of serialized Model object")
@plac.opt(
    "output",
    help="How to output the plot. 1: show on screen, 2: save to file, 3: both 1 and 2.",
    choices=[1, 2, 3],
)
@plac.opt(
    "fig_fname",
    abbrev="fn",
    help="filepath to save generated figure file if output is 2 or 3",
)
@plac.flg(
    "mark_min",
    abbrev="mm",
    help="""
        Mark data points with lowest property value. This is useful, for instance, 
        to indicate ground-state structures if the property is the energy.
    """,
)
@plac.flg(
    "show_loo_predictions",
    abbrev="loo",
    help="""
        Show test predictions from CV - LeaveOneOut approach.
    """,
)
def plot_property(
    sset_filepath: str,
    property_name: str,
    sset_gss_filepath: Optional[str] = None,
    model_filepath: Optional[str] = None,
    output: int = 1,
    fig_fname: Optional[str] = None,
    mark_min: bool = False,
    show_loo_predictions: bool = False,
):
    """Make plot of property versus concentration"""
    cmd_message("head")

    show_plot = True
    if output == 2:
        show_plot = False

    if output != 1:
        if fig_fname is None:
            fig_fname = "plot_property.png"

    sset = StructuresSet(filepath=sset_filepath)
    sset_gss = (
        StructuresSet(filepath=sset_gss_filepath)
        if sset_gss_filepath is not None
        else None
    )

    model = Model(filepath=model_filepath) if model_filepath is not None else None

    plot_property_vs_concentration(
        sset,
        property_name,
        cemodel=model,
        show_plot=show_plot,
        fig_fname=fig_fname,
        show_loo_predictions=show_loo_predictions,
        sset_gss=sset_gss,
    )
