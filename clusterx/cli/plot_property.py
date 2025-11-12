# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.model import Model
from clusterx.structures_set import StructuresSet
from clusterx.visualization import plot_property_vs_concentration

commands = ["plot_property"]


@plac.annotations(
    sset_filepath=(
        "Path to file of serialized StructuresSet object.",
        "positional",
        None,
        str,
    ),
    property_name=("Name of the property to be plotted.", "positional", None, str),
    sset_gss_filepath=(
        "Path to file of serialized StructuresSet object for GS structures.",
        "option",
        "gss",
        str,
    ),
    model_filepath=("Path to file of serialized Model object.", "option", None, str),
    output=("How to output the plot. 1: show, 2: save, 3: both.", "option", None, int),
    fig_fname=(
        "Filepath to save generated figure if output is 2 or 3.",
        "option",
        "fn",
        str,
    ),
    figdata_fname=(
        "Filepath to save the data used to generate the figure. Possible extensions: .npz, .json, .txt, .dat.",
        "option",
        "fdn",
        str,
    ),
    mark_min=(
        "Mark points with the lowest property value (e.g. ground states).",
        "flag",
        "mm",
        bool,
    ),
    show_loo_predictions=(
        "Show Leave-One-Out cross-validation predictions.",
        "flag",
        "loo",
        bool,
    ),
)
def plot_property(
    sset_filepath: str,
    property_name: str,
    sset_gss_filepath: Optional[str] = None,
    model_filepath: Optional[str] = None,
    output: int = 1,
    fig_fname: Optional[str] = None,  # possible extensions: .npz, .json, .txt, .dat
    figdata_fname: Optional[str] = None,
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
        data_fname=figdata_fname,
        show_loo_predictions=show_loo_predictions,
        sset_gss=sset_gss,
    )
