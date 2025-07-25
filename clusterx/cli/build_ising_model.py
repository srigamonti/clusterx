# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import plac
from ase import Atoms
from sklearn.linear_model import LinearRegression

from clusterx.cli.config_utils import cmd_message
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell

commands = ["build_ising_model"]


@plac.annotations(
    e0=("Constant coefficient (intercept)", "option", "e0", float),
    h0=("External field ", "option", "h0", float),
    j1=("Spin-spin interaction (first neighbor)", "option", "j1", float),
    suffix=(
        "Suffix to be added to the file names created by this command",
        "option",
        None,
        str,
    ),
    prefix=(
        "Prefix to be added to the file names created by this command",
        "option",
        None,
        str,
    ),
)
def build_ising_model(
    e0: float = 0.0,
    h0: float = 0.0,
    j1: float = 1.0,
    suffix: str = "",
    prefix: str = "",
):
    """Build 2D-Ising model. Used for demonstration purposes"""
    cmd_message("head")

    a = 1.0
    positions = [(0, 0, 0)]
    cell = [(a, 0, 0), (0, a, 0), (0, 0, a)]
    pbc = (True, True, False)

    pri = Atoms("H", positions=positions, cell=cell, pbc=pbc)
    sub = Atoms("He", positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)

    plat.serialize(filepath=f"{prefix}plat-ising{suffix}.json")

    cpool = ClustersPool(parent_lattice=plat, npoints=[1, 2], radii=[0, 1.1])

    cpool.serialize(filepath=f"{prefix}cpool-ising{suffix}.json")

    ccalc = CorrelationsCalculator(basis_name="chebyshev", parent_lattice=plat, clusters_pool=cpool)

    estimator = LinearRegression()
    n_features = len(cpool)

    estimator.coef_ = np.array([-h0, -j1])
    estimator.intercept_ = e0

    cemodel = Model(corrc=ccalc, property_name="Energy", estimator=estimator)

    cemodel.serialize(filepath=f"{prefix}ising-model{suffix}.pickle")
