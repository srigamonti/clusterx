# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import math

import pytest
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as cn

from clusterx.test.defaults import get_clathrate_plat
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.clusters.cluster import Cluster
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.thermodynamics.wang_landau import WangLandau
from clusterx.cli.wang_landau import wang_landau


@pytest.fixture
def plat():
    # parent lattice with single site and one substitution
    cell = [1, 1, 1]
    positions = [[0, 0, 0]]
    pbc = [True, True, False]

    pri = Atoms(["Cu"], positions=positions, cell=cell, pbc=pbc)
    sub = Atoms(["Au"], positions=positions, cell=cell, pbc=pbc)
    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)
    return plat


@pytest.fixture
def cpool(plat):
    # one-point clusters and nearest neighbor two-point clusters
    return ClustersPool(plat, npoints=[1, 2], radii=[0, 1.1])


@pytest.fixture
def model(plat, cpool):
    corc = CorrelationsCalculator("trigonometric", plat, cpool)
    ecisE = [0., -1.]
    multT = cpool.get_multiplicities()
    cemodel = Model(corc, "energy", ecis=np.multiply(ecisE, multT))
    return cemodel


def test_cli(plat, model):
    # serialize objects
    model_filepath = "model_wl.pickle"
    model.serialize(model_filepath)
    plat_filepath = "plat_wl.json"
    plat.serialize(plat_filepath)

    sc_shape = [8, 8]
    nsubs = {0: [int(np.prod(sc_shape) / 2)]}  # one substitution for the whole supercell
    energy_range = [-2., 3.]

    wang_landau(
        nsubs=nsubs,
        model_filepath=model_filepath,
        plat_filepath=plat_filepath,
        sc_shape=sc_shape,
        energy_range=energy_range,
    )


def test_sampling_clathrate():
    np.random.seed(
        10
    )  # setting a seed for the random package for comparible random structures

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
    cpool.add_cluster(Cluster([35, 11], [s, s], cpsc))
    cpool.add_cluster(Cluster([39, 30], [s, s], cpsc))
    cpool.add_cluster(Cluster([22, 17], [s, s], cpsc))
    cpool.add_cluster(Cluster([35, 42], [s, s], cpsc))
    cpool.add_cluster(Cluster([32, 14], [s, s], cpsc))
    cpool.add_cluster(Cluster([11, 10], [s, s], cpsc))
    cpool.add_cluster(Cluster([18, 9], [s, s], cpsc))
    cpool.add_cluster(Cluster([18, 43], [s, s], cpsc))

    # Energy
    cpoolE = cpool.get_subpool([0, 1, 2, 3, 4, 5, 6, 7, 9, 15])
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

    corcE = CorrelationsCalculator("binary-linear", plat, cpoolE)
    p = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    scell = SuperCell(plat, p)
    nsubs = {0: [16]}
    cemodelE = Model(corcE, "energy", ecis=np.multiply(ecisE, multT))

    wl = WangLandau(
        energy_model=cemodelE, scell=scell, ensemble="canonical", nsubs=nsubs
    )
    e0_unitcell = -77652.707924876348
    e0 = float(e0_unitcell)
    e1 = e0 + 0.5
    cdos = wl.wang_landau_sampling(
        energy_range=[e0, e1],
        energy_bin_width=0.002,
        f_range=[math.exp(1), 2],
        update_method="square_root",
        flatness_conditions=[[0.1, math.exp(1e-1)]],
    )
    energy_bins, gs = cdos.get_cdos(ln=True, normalization=False)

    # renergy_bins = [-77652.64792487655, -77652.64592487656, -77652.64192487657, -77652.6339248766, -77652.63192487661, -77652.62992487662, -77652.62792487662, -77652.62592487663, -77652.62392487664, -77652.62192487664, -77652.61992487665, -77652.61792487666, -77652.61592487666, -77652.61392487667, -77652.61192487668, -77652.60992487668, -77652.60792487669, -77652.6059248767, -77652.6039248767, -77652.60192487671, -77652.59992487672, -77652.59792487673, -77652.59592487673, -77652.59392487674, -77652.59192487675, -77652.58992487675, -77652.58792487676, -77652.58592487677, -77652.58392487677, -77652.58192487678, -77652.57992487679, -77652.5779248768, -77652.5759248768, -77652.57392487681, -77652.57192487681, -77652.56992487682, -77652.56792487683, -77652.56592487684, -77652.56392487684, -77652.56192487685, -77652.55992487686, -77652.55792487686, -77652.55592487687, -77652.55392487688, -77652.55192487688, -77652.54992487689, -77652.5479248769, -77652.5459248769, -77652.54392487691, -77652.54192487692, -77652.53992487692, -77652.53792487693, -77652.53592487694, -77652.53392487695, -77652.53192487695, -77652.52992487696, -77652.52792487697, -77652.52592487697, -77652.52392487698, -77652.52192487699, -77652.519924877, -77652.517924877, -77652.515924877, -77652.51392487701, -77652.50992487703, -77652.50792487703, -77652.50392487705, -77652.49392487708, -77652.4879248771]
    # rgs = [3.0, 2.0, 3.0, 2.0, 6.0, 4.0, 5.0, 10.0, 3.0, 8.0, 8.0, 9.0, 8.0, 9.0, 10.0, 7.0, 10.0, 10.0, 8.0, 13.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 11.0, 8.0, 12.0, 12.0, 13.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 12.0, 13.0, 11.0, 11.0, 11.0, 11.0, 11.0, 10.0, 8.0, 9.0, 9.0, 10.0, 8.0, 12.0, 9.0, 9.0, 9.0, 3.0, 5.0, 8.0, 8.0, 3.0, 6.0, 9.0, 3.0, 5.0, 8.0, 2.0, 7.0, 2.0, 2.0, 2.0]
    # np.testing.assert_allclose(energy_bins, renergy_bins, rtol=1e-4)
    # np.testing.assert_allclose(gs, rgs, rtol=1e-4)
