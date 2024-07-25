# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac
from clusterx.correlations import CorrelationsCalculator
from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.cli.config_utils import cmd_message
from clusterx.model import ModelBuilder

commands = ["build_model"]


@plac.pos(
    "property_name",
    help="Property to be modelled. Must be present in the StructuresSet object.",
)
@plac.opt(
    "ccalc_filepath",
    abbrev="ccf",
    help="Path to the pickle file of a serialized CorrelationsCalculator object.",
)
@plac.opt(
    "sset_filepath",
    abbrev="ssf",
    help=("Path to a serialized StructuresSet object.",),
)
@plac.opt(
    "cpool_filepath",
    abbrev="ssf",
    help=("Path to a serialized ClustersPool object.",),
)
@plac.opt(
    "model_filepath",
    abbrev="mof",
    help=("Path to serialize the created Model object.",),
)
@plac.opt(
    "selector_type",
    abbrev="st",
    help=("Selector type.",),
)
@plac.opt(
    "selector_opts",
    abbrev="so",
    help=("Selector options.",),
)
@plac.opt(
    "estimator_type",
    abbrev="et",
    help=("Estimator type.",),
)
@plac.opt(
    "estimator_opts",
    abbrev="eo",
    help=("Estimator options.",),
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
):
    """Compute CE model"""
    cmd_message("head")

    ccalc = CorrelationsCalculator(filepath=ccalc_filepath)
    sset = StructuresSet(filepath=sset_filepath)
    cpool = ClustersPool(filepath=cpool_filepath)

    mb = ModelBuilder(
        selector_type=selector_type,
        selector_opts=selector_opts,
        estimator_type=estimator_type,
        estimator_opts=estimator_opts,
    )

    model = mb.build(sset, cpool, property_name, corrc=ccalc)

    model.serialize(filepath=model_filepath)
