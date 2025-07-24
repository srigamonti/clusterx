# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac
from ase import Atoms

from clusterx.cli.config_utils import cmd_message
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell

commands = ["build_ising_model"]


@plac.annotations(
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
    true3d=(
        "By default 2D Ising model is built. Put this to true for 3D Ising model",
        "flag",
        "t3d",
        bool,
    ),
)
def build_ising_model(suffix: str = "", prefix: str = "", true3d: bool = False):
    """Build Ising model. Used for demonstration purposes"""
    cmd_message("head")

    a = 1.0
    positions = [(0, 0, 0)]
    cell = [(a, 0, 0), (0, a, 0), (0, 0, a)]
    if not true3d:
        pbc = (True, True, False)
    else:
        raise NotImplementedError("3D Ising is not implemented yet.")

    pri = Atoms("H", positions=positions, cell=cell, pbc=pbc)
    sub = Atoms("He", positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)

    if not true3d:
        plat.serialize(filepath=f"{prefix}plat-ising-2d{suffix}.json")
    else:
        plat.serialize(filepath=f"{prefix}plat-ising-3d{suffix}.json")

    scell = SuperCell(plat, p=2)
    scell.serialize(filepath=f"{prefix}scell-ising-3d{suffix}.json")

    cpool = ClustersPool(
        parent_lattice=plat, npoints=[2], radii=[1.1], super_cell=scell
    )
    cpool.gen_clusters()

    if not true3d:
        cpool.serialize(filepath=f"{prefix}cpool-ising-2d{suffix}.json")
    else:
        cpool.serialize(filepath=f"{prefix}cpool-ising-3d{suffix}.json")

    ccalc = CorrelationsCalculator(
        basis="chebyshev", parent_lattice=plat, clusters_pool=cpool
    )

    """
    scell = SuperCell(plat, p=2)
    cpool = ClustersPool(plat, npoints=[1, 2, 3], radii=[0, -1, -1], super_cell=scell)
    ccalc = CorrelationsCalculator(basis="indicator-binary", parent_lattice=plat, clusters_pool=cpool)
    estimator = LinearRegression()
    n_features = len(cpool)

    random_coefficients = rng.random(size=n_features)
    random_coefficients -= random_coefficients.mean()
    estimator.coef_ = random_coefficients
    estimator.intercept_ = 0

    cemodel = Model(corrc=ccalc, property_name=property_name, estimator=estimator)

    cemodel.serialize(filepath=model_filepath)
    """
