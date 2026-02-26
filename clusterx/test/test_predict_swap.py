# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import random

import pytest
from ase import Atoms
from ase.build import bulk
import numpy as np

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.model import Model
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.thermodynamics.monte_carlo import MonteCarlo
from clusterx.test.defaults import get_clathrate_supercell, get_clathrate_model


@pytest.fixture
def plat_cubic():
    pri = bulk("H", crystalstructure="sc", a=1.0)
    sub = bulk("He", crystalstructure="sc", a=1.0)
    return ParentLattice(atoms=pri, substitutions=[sub], pbc=(1, 1, 1))


@pytest.fixture
def cpool_cubic(plat_cubic):
    return ClustersPool(plat_cubic, npoints=[1, 2], radii=[0, 1.1])


@pytest.fixture
def corrc_cubic(plat_cubic, cpool_cubic):
    return CorrelationsCalculator("trigonometric", plat_cubic, cpool_cubic)


@pytest.fixture
def model_cubic(plat_cubic, cpool_cubic, corrc_cubic):
    mult = cpool_cubic.get_multiplicities()

    corrc = CorrelationsCalculator("trigonometric", plat_cubic, cpool_cubic)
    model = Model(corrc, "energy2", ecis=mult)
    model.reset_mc(True)
    return model


def seed_rngs(seed=0):
    random.seed(seed)
    np.random.seed(seed)


# test cases (TODO):
# estimator: None (just ECI), linear, non-linear
@pytest.mark.parametrize("basis", ["binary-linear", "trigonometric", "polynomial"])
def test_swap_clathrate(basis):
    seed_rngs(42)  # is specific, so that swap changes energy
    scell = get_clathrate_supercell()
    model = get_clathrate_model(basis)
    model.reset_mc(True)

    structure = scell.gen_random_structure()
    i0, i1 = 0, 1
    e0 = model.predict(structure)
    e_diff_swap = model.predict_swap(structure, i0, i1)
    structure.swap(i0, i1)
    e1 = model.predict(structure)
    np.testing.assert_allclose(e1 - e0, e_diff_swap)


def test_swap(plat_cubic, model_cubic):
    seed_rngs(42)

    p = [3, 3, 3]  # for 2*2*2 supercell, results are unreliable due to wrapping
    scell = SuperCell(plat_cubic, p)
    structure = scell.gen_random_structure(nsubs=int(np.prod(p) / 2))
    preds_full = []
    preds_swap = []
    for i in range(len(scell)):
        for j in range(len(scell)):
            if j > i:
                continue
            pred_init = model_cubic.predict(structure)
            pred_swap = model_cubic.predict_swap(structure, i, j)
            structure.swap(i, j)
            pred_final = model_cubic.predict(structure)
            preds_full.append(pred_final - pred_init)
            preds_swap.append(pred_swap)
    np.testing.assert_allclose(preds_swap, preds_full, atol=1e-10)


@pytest.mark.parametrize("reduce", [False, True])
def test_flip_2D(reduce):
    seed_rngs(42)
    pri = Atoms("Cu", positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=[1, 1, 0])
    sub = Atoms("Au", positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=[1, 1, 0])
    plat = ParentLattice(atoms=pri, substitutions=[sub])
    p = [10, 10, 1]
    scell = SuperCell(plat, p)
    structure = scell.gen_random_structure(nsubs=int(np.prod(p) / 2))
    cpool = ClustersPool(plat, npoints=[2], radii=[1.5])
    corrc = CorrelationsCalculator("trigonometric", plat, cpool)
    model = Model(corrc, "energy", ecis=[1, 1])
    if reduce:
        model.init_reduced_model(structure)
        model._init_interaction_dict(model.scell_reduced)
        structure.init_reduced_structure(model.scell_reduced)

    preds_full = []
    preds_flip = []
    for i in range(len(structure)):
        pred_init = model.predict(structure)
        old_sigma = structure.sigmas[i]
        new_sigma = 1 - old_sigma
        pred_flip = model.predict_flip(structure, i, old_sigma, new_sigma, reduce)
        structure.sigmas[i] = new_sigma
        pred_final = model.predict(structure)
        preds_full.append(pred_final - pred_init)
        preds_flip.append(pred_flip)
    np.testing.assert_allclose(preds_flip, preds_full, atol=1e-12)


@pytest.mark.parametrize("reduce", [False, True])
def test_flip(plat_cubic, model_cubic, reduce):
    seed_rngs(42)

    p = [5, 5, 5]
    scell = SuperCell(plat_cubic, p)
    structure = scell.gen_random_structure(nsubs=int(np.prod(p) / 2))
    if reduce:
        model_cubic.init_reduced_model(structure)
        model_cubic._init_interaction_dict(model_cubic.scell_reduced)
        structure.init_reduced_structure(model_cubic.scell_reduced)
    preds_full = []
    preds_flip = []
    for i in range(len(structure)):
        pred_init = model_cubic.predict(structure)
        old_sigma = structure.sigmas[i]
        new_sigma = 1 - old_sigma
        pred_flip = model_cubic.predict_flip(structure, i, old_sigma, new_sigma, reduce)
        structure.sigmas[i] = new_sigma
        pred_final = model_cubic.predict(structure)
        preds_full.append(pred_final - pred_init)
        preds_flip.append(pred_flip)
    np.testing.assert_allclose(preds_flip, preds_full)


def test_predict_partial_flip(plat_cubic, model_cubic):
    model = model_cubic
    seed_rngs(42)
    p = [5, 5, 5]
    scell = SuperCell(plat_cubic, p)
    structure = scell.gen_random_structure(nsubs=int(np.prod(p) / 2))

    model.init_reduced_model(structure)
    model._init_interaction_dict(model.scell_reduced)
    structure.init_reduced_structure(model.scell_reduced)

    preds_full = []
    preds_partial = []
    for i in range(len(structure)):
        pred_init = model.predict(structure)
        pred_partial_i = model.predict_partial(structure, i)
        old_sigma = structure.sigmas[i]
        new_sigma = 1 - old_sigma

        structure.sigmas[i] = new_sigma
        pred_final = model.predict(structure)
        pred_partial_f = model.predict_partial(structure, i)
        preds_full.append(pred_final - pred_init)
        preds_partial.append(pred_partial_f - pred_partial_i)
    np.testing.assert_allclose(preds_partial, preds_full)


def test_flip_scale_sc(plat_cubic, model_cubic):
    for i in range(3, 10):
        model_cubic.reset_mc(True)
        p = [3, 3, 3 * i]
        scell = SuperCell(plat_cubic, p)
        structure = scell.get_pristine_structure()
        pred_init = model_cubic.predict(structure)
        old_sigma = structure.sigmas[0]
        new_sigma = 1 - old_sigma
        pred_flip = model_cubic.predict_flip(structure, 0, old_sigma, new_sigma)
        structure.sigmas[i] = new_sigma
        pred_final = model_cubic.predict(structure)
        np.testing.assert_allclose(pred_flip, pred_final - pred_init)


def test_metropolis_cubic(plat_cubic, model_cubic):
    seed = 42
    p = [4, 4, 4]
    scell = SuperCell(plat_cubic, p)
    nsubs = {0: [4]}
    nsteps = 10  # might fail for larger nsteps and similar or smaller p

    mc_full = MonteCarlo(
        energy_model=model_cubic,
        scell=scell,
        nsubs=nsubs,
        predict_swap=False,
        seed=seed,
    )
    traj_full = mc_full.metropolis(temperature=100, no_of_sampling_steps=nsteps)

    mc_swap = MonteCarlo(
        energy_model=model_cubic, scell=scell, nsubs=nsubs, predict_swap=True, seed=seed
    )
    traj_swap = mc_swap.metropolis(temperature=100, no_of_sampling_steps=nsteps)
    np.testing.assert_allclose(traj_swap.get_energies(), traj_full.get_energies())
