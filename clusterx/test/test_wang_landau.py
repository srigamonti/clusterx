# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import os
import math
import random

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
    ecisE = [0.0, -1.0]
    multT = cpool.get_multiplicities()
    cemodel = Model(corc, "energy", ecis=np.multiply(ecisE, multT))
    return cemodel


@pytest.mark.parametrize("reduce_structure", [False, True])
def test_reduced(plat, model, reduce_structure):
    p = [4, 4, 1]
    scell = SuperCell(plat, p)
    nsubs = {0: [int(np.prod(p) / 2)]}
    energy_range = [-2.0, 3.0]

    wl = WangLandau(
        energy_model=model,
        scell=scell,
        ensemble="canonical",
        nsubs=nsubs,
        predict_swap=True,
        reduce_structure=reduce_structure,
    )
    cdos = wl.wang_landau_sampling(
        energy_range=energy_range,
        energy_bin_width=0.2,
        f_range=[4, 1.9],
        update_method="square_root",
        flatness_conditions=[[0.5, 2], [0.5, 1]],
        initial_decoration=None,
        serialize=False,
        filename="cdos.json",
        serialize_during_sampling=False,
        restart_from_file=False,
        plot_hist_real_time=False,
        acc_prob_init_structure=1e-3,
        acc_prob_dist_init_structure="gaussian",
        itmax_init_structure=int(1e8),
        niter_per_sweep=1000,
        nproc=0,
        seed=1,
    )
    cdos.get_cdos(ln=True, normalization=False)


def test_cli(plat, model):
    # serialize objects
    model_filepath = "model_wl.pickle"
    model.serialize(model_filepath)
    plat_filepath = "plat_wl.json"
    plat.serialize(plat_filepath)

    sc_shape = [4, 4]
    nsubs = {0: [int(np.prod(sc_shape) / 2)]}  # one substitution for the whole supercell
    energy_range = [-2.0, 3.0]

    wang_landau(
        plat_filepath=plat_filepath,
        model_filepath=model_filepath,
        sc_shape=sc_shape,
        nsubs=nsubs,
        ensemble="canonical",
        sublattice_indices=[],
        chemical_potentials=None,
        predict_swap=False,
        error_reset=None,
        energy_range=energy_range,
        energy_bin_width=0.2,
        f_range=[4, 1.9],
        update_method="square_root",
        flatness_conditions=[[0.5, 2], [0.5, 1]],
        initial_decoration=None,
        serialize=True,
        filename="cdos.json",
        serialize_during_sampling=True,
        restart_from_file=False,
        plot_hist_real_time=False,
        acc_prob_init_structure=1e-3,
        acc_prob_dist_init_structure="gaussian",
        itmax_init_structure=int(1e8),
        niter_per_sweep=1000,
        nproc=0,
        seed=1,
    )
    assert os.path.exists("cdos.json")


def test_sampling_clathrate():
    np.random.seed(10)  # setting a seed for the random package for comparible random structures
    random.seed(45)

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

    # Energy
    ecis = [
        -78407.3247588,
        47.164484875,
        47.1673476881,
        47.1569012692,
        0.00851281608144,
        0.0139835351147,
    ]
    multiplicities = [1, 24, 16, 6, 12, 8]

    corcE = CorrelationsCalculator("binary-linear", plat, cpool)
    p = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    scell = SuperCell(plat, p)
    nsubs = {0: [16]}
    cemodelE = Model(corcE, "energy", ecis=np.multiply(ecis, multiplicities))

    wl = WangLandau(
        energy_model=cemodelE,
        scell=scell,
        ensemble="canonical",
        nsubs=nsubs,
        predict_swap=False,
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
        plot_hist_real_time=False,
        niter_per_sweep=10000,
        seed=11,  # this seed converges within one sweep
    )
    energy_bins, gs = cdos.get_cdos(ln=True, normalization=False)
