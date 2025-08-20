# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from typing import Optional

import plac
import numpy as np
from clusterx.cli.config_utils import cmd_message, get_command_name
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import ModelBuilder
from clusterx.structures_set import StructuresSet

commands = ["build_model"]


@plac.annotations(
    property_name=("Property to be modelled.", "positional", None, str),
    ccalc_filepath=("Path to CorrelationsCalculator object.", "option", "ccf", str),
    sset_filepath=("Path to StructuresSet object.", "option", "ssf", str),
    cpool_filepath=("Path to ClustersPool object.", "option", "cpf", str),
    model_filepath=("Output Model file path.", "option", "mof", str),
    selector_type=("Selector type.", "option", "st", str),
    selector_opts=("Selector options.", "option", "so", dict),
    estimator_type=("Estimator type.", "option", "et", str),
    estimator_opts=("Estimator options.", "option", "eo", dict),
    plot_optimization_vs_sparsity=("Plot config.", "option", "plotovsd", dict),
    weights_filepath=("Path to npz file containing weights.", "option", "wf", str),
)
def build_model(
    property_name: str,
    ccalc_filepath: str = "ccalc.pickle",
    sset_filepath: str = "sset.json",
    cpool_filepath: str = "cpool.json",
    model_filepath: str = "model.pickle",
    selector_type: str = "identity",
    selector_opts: dict = {"fit_intercept": True},
    estimator_type: str = "skl_LinearRegression",
    estimator_opts: dict = {"fit_intercept": True},
    plot_optimization_vs_sparsity: Optional[dict] = None,
    weights_filepath: Optional[str] = None,
):
    """Compute CE model"""
    cmd_message("head")

    print(f"Info({get_command_name()}): Initialization")
    ccalc = CorrelationsCalculator(filepath=ccalc_filepath)
    sset = StructuresSet(filepath=sset_filepath)
    cpool = ClustersPool(filepath=cpool_filepath)

    if weights_filepath is not None:
        weights = np.load(weights_filepath)["weights"]
        kwargs = {"sample_weight": weights}
    else:
        kwargs = {}

    mb = ModelBuilder(
        selector_type=selector_type,
        selector_opts=selector_opts,
        estimator_type=estimator_type,
        estimator_opts=estimator_opts,
    )

    print(f"Info({get_command_name()}): Computing model")
    model = mb.build(sset, cpool, property_name, corrc=ccalc, **kwargs)

    if plot_optimization_vs_sparsity is not None:
        print(
            f"Info({get_command_name()}): Generating plot of optimization vs sparsity"
        )

        from clusterx.visualization import plot_optimization_vs_sparsity as povs

        povs(
            mb.get_selector(),
            show_plot=plot_optimization_vs_sparsity["show_plot"],
            fname=plot_optimization_vs_sparsity["filepath"],
        )

    print(f"Info({get_command_name()}): Updating model and ccalc")

    model.serialize(filepath=model_filepath)
    ccalc.serialize(filepath=ccalc_filepath)
