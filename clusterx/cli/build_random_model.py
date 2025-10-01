# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
import random
from typing import Optional

import numpy as np
import plac
from sklearn.linear_model import LinearRegression

from clusterx.cli.config_utils import cmd_message
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell

commands = ["build_random_model"]


@plac.annotations(
    plat_filepath=(
        "Filepath of a serialized ParentLattice object.",
        "option",
        "pl",
        str,
    ),
    property_name=(
        "Property to be modelled. Must be present in the StructuresSet object.",
        "option",
        None,
        str,
    ),
    model_filepath=("Path to serialize the created Model object.", "option", None, str),
    random_seed=(
        "Seed to initialize the random generators for reproducibility.",
        "option",
        None,
        int,
    ),
)
def build_random_model(
    plat_filepath: Optional[str] = None,
    property_name: str = "random_property",
    model_filepath: str = "random_model.pickle",
    random_seed: Optional[int] = None,
):
    """Build a random model given a parent lattice. Used for demonstration purposes"""
    cmd_message("head")

    random.seed(random_seed)
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    plat = ParentLattice(filepath=plat_filepath)
    scell = SuperCell(plat, p=2)
    cpool = ClustersPool(plat, npoints=[1, 2, 3], radii=[0, -1, -1], super_cell=scell)
    ccalc = CorrelationsCalculator(
        basis="indicator-binary", parent_lattice=plat, clusters_pool=cpool
    )
    estimator = LinearRegression()
    n_features = len(cpool)

    random_coefficients = rng.random(size=n_features)
    random_coefficients -= random_coefficients.mean()
    estimator.coef_ = random_coefficients
    estimator.intercept_ = 0

    cemodel = Model(corrc=ccalc, property_name=property_name, estimator=estimator)

    cemodel.serialize(filepath=model_filepath)
