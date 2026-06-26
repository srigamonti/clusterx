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


@pytest.fixture
def plat_perov():
    """Perovskite mult-sublattice parent lattice"""
    cell = [6, 6, 6]
    positions = [
        [0, 0, 0],
        [3, 3, 3],
        [0, 3, 3],
        [3, 0, 3],
        [3, 3, 0],
    ]
    pbc = [True, True, True]
    pri = Atoms(["Cs", "Pb", "I", "I", "I"], positions=positions, cell=cell, pbc=pbc)
    sub_Sn = Atoms(["Cs", "Sn", "I", "I", "I"], positions=positions, cell=cell, pbc=pbc)
    sub_Br = Atoms(["Cs", "Pb", "Br", "Br", "Br"], positions=positions, cell=cell, pbc=pbc)
    plat = ParentLattice(pri, substitutions=[sub_Sn, sub_Br], pbc=pbc)
    return plat


@pytest.fixture
def cpool_perov(plat_perov):
    return ClustersPool(plat_perov, npoints=[1, 2], radii=[0, 6.1])


@pytest.fixture
def corrc_perov(plat_perov, cpool_perov):
    return CorrelationsCalculator("trigonometric", plat_perov, cpool_perov)


@pytest.fixture
def model_perov(cpool_perov, corrc_perov):
    ecis = np.array([0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337, -0.23413696, 1.57921282])
    return Model(corrc_perov, "energy", ecis=ecis)


def test_perov(plat_perov, model_perov):
    p = [4, 4, 4]
    plat_perov.print_sublattice_types()
    scell = SuperCell(plat_perov, p)
    nsubs = {0: [int(len(scell) / 5 * 3 / 2)], 2: [int(len(scell) / 5 / 2)]}

    wl = WangLandau(
        energy_model=model_perov,
        scell=scell,
        nsubs=nsubs,
        predict_swap=True,
    )
    cdos = wl.wang_landau_sampling(
        energy_range=[-2.0, 3.0],
        energy_bin_width=0.2,
        f_range=[4, 1.9],
        flatness_conditions=[[0.5, 2], [0.5, 1]],
        niter_per_sweep=1000,
        nproc=0,
        seed=1,
    )
    cdos_energies, cdos_weights = cdos.get_cdos(ln=True, normalization=False)
    print(np.array(cdos_weights))


def test_nonlinear(plat_perov, corrc_perov):
    """Test that WL returns same CDOS with predict_swap False and True, using a
    nonlinear model."""
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
    n_corrs = len(corrc_perov.get_cpool())
    np.random.seed(42)
    X = np.random.uniform(low=-1, high=1, size=(100, n_corrs))
    y = np.random.uniform(low=-1, high=1, size=(100,))
    pipeline.fit(X, y)
    model = Model(corrc_perov, "energy", estimator=pipeline)

    p = [4, 4, 4]
    scell = SuperCell(plat_perov, p)
    nsubs = {0: [int(len(scell) / 5 * 3 / 2)], 2: [int(len(scell) / 5 / 2)]}

    cdos_compare = []
    for predict_swap in [False, True]:
        wl = WangLandau(
            energy_model=model,
            scell=scell,
            nsubs=nsubs,
            predict_swap=predict_swap,
        )
        cdos = wl.wang_landau_sampling(
            energy_range=[-0.1, 0.0],
            energy_bin_width=0.01,
            f_range=[4, 1.9],
            flatness_conditions=[[0.5, 2], [0.5, 1]],
            niter_per_sweep=10,
            nproc=0,
            seed=1,
            nonlinear=True,
        )
        _, cdos_weights = cdos.get_cdos(ln=True, normalization=False)
        cdos_compare.append(np.array(cdos_weights))
    np.testing.assert_allclose(cdos_compare[0], cdos_compare[1])


def test_swap(plat_perov, model_perov):
    """Test that WL returns same CDOS with predict_swap False and True."""
    p = [4, 4, 4]
    scell = SuperCell(plat_perov, p)
    nsubs = {0: [int(len(scell) / 5 * 3 / 2)], 2: [int(len(scell) / 5 / 2)]}

    cdos_compare = []
    for predict_swap in [False, True]:
        wl = WangLandau(
            energy_model=model_perov,
            scell=scell,
            nsubs=nsubs,
            predict_swap=predict_swap,
        )
        cdos = wl.wang_landau_sampling(
            energy_range=[-0.1, 0.0],
            energy_bin_width=0.01,
            f_range=[4, 1.9],
            flatness_conditions=[[0.5, 2], [0.5, 1]],
            niter_per_sweep=1000,
            nproc=0,
            seed=1,
        )
        _, cdos_weights = cdos.get_cdos(ln=True, normalization=False)
        cdos_compare.append(np.array(cdos_weights))
    np.testing.assert_allclose(cdos_compare[0], cdos_compare[1])


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
