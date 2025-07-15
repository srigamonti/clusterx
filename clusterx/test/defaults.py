"""Default objects for testing purposes."""

import numpy as np
from ase.spacegroup import crystal
from ase.data import atomic_numbers as cn

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.cluster import Cluster
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model


def get_clathrate_plat():
    a = 10.5148
    x = 0.185
    y = 0.304
    z = 0.116
    wyckoff = [
        (0, y, z),  # 24k
        (x, x, x),  # 16i
        (1 / 4.0, 0, 1 / 2.0),  # 6c
        (1 / 4.0, 1 / 2.0, 0),  # 6d
        (0, 0, 0),  # 2a
    ]

    # Build the parent lattice
    pri = crystal(
        ["Si", "Si", "Si", "Ba", "Ba"],
        wyckoff,
        spacegroup=223,
        cellpar=[a, a, a, 90, 90, 90],
    )
    sub = crystal(
        ["Al", "Al", "Al", "Ba", "Ba"],
        wyckoff,
        spacegroup=223,
        cellpar=[a, a, a, 90, 90, 90],
    )
    plat = ParentLattice(atoms=pri, substitutions=[sub])
    return plat


def get_clathrate_supercell(p_cell=[(2, 0, 0), (0, 2, 0), (0, 0, 2)]):
    plat = get_clathrate_plat()
    return SuperCell(plat, p=p_cell)


def get_clathrate_cpool():
    plat = get_clathrate_plat()
    cpool = ClustersPool(plat)
    cpsc = cpool.get_cpool_scell()
    s = cn["Al"]
    cpool.add_cluster(Cluster([], [], cpsc))
    cpool.add_cluster(Cluster([0], [s], cpsc))
    cpool.add_cluster(Cluster([24], [s], cpsc))
    cpool.add_cluster(Cluster([40], [s], cpsc))
    cpool.add_cluster(Cluster([6, 4], [s, s], cpsc))
    cpool.add_cluster(Cluster([37, 32], [s, s], cpsc))
    cpool.add_cluster(Cluster([39, 12], [s, s], cpsc))
    cpool.add_cluster(Cluster([16, 43], [s, s], cpsc))
    cpool.add_cluster(Cluster([39, 30], [s, s], cpsc))
    cpool.add_cluster(Cluster([18, 43], [s, s], cpsc))
    return cpool


def get_clathrate_corrcalc(basis):
    plat = get_clathrate_plat()
    cpool = get_clathrate_cpool()
    corrcalc = CorrelationsCalculator(basis, plat, cpool)
    return corrcalc


def get_clathrate_model(basis):
    corrcalc = get_clathrate_corrcalc(basis)
    ecisE = [
        -78407.3247588,
        47.164484875,
        47.1673476881,
        47.1569012692,
        0.00851281608144,
        0.0139835351147,
        0.0108175321899,
        0.0101521144776,
        0.00121744613474,
        0.000413664306204,
    ]
    multT = [1, 24, 16, 6, 12, 8, 48, 24, 24, 24]
    model = Model(corrcalc, "energy", ecis=np.multiply(ecisE, multT))
    return model
