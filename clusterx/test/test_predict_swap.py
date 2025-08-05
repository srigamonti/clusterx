# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import random

import pytest
from ase.build import bulk
import numpy as np

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet
from clusterx.model import Model, ModelBuilder
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.calculators.emt import EMT2
from clusterx.correlations import CorrelationsCalculator
from clusterx.thermodynamics.monte_carlo import MonteCarlo
from clusterx.test.defaults import (
    get_clathrate_supercell,
    get_clathrate_model
)


def seed_rnd_generator(seed=0):
    random.seed(seed)
    np.random.seed(seed)

# test cases (TODO):
# estimator: None (just ECI), linear, non-linear
@pytest.mark.parametrize('basis', ['binary-linear', 'trigonometric', 'polynomial'])
def test_clathrate_swap(basis):
    seed_rnd_generator(42) # is specific, so that swap changes energy
    scell = get_clathrate_supercell()
    model = get_clathrate_model(basis)
    model.reset_mc(True)

    structure = scell.gen_random_structure()
    i0, i1 = 0, 1
    e0 = model.predict(structure)
    e_diff_swap = model.predict_swap(structure, i0, i1)
    structure.swap(i0, i1)
    e1 = model.predict(structure)
    np.testing.assert_allclose(e0 - e1, e_diff_swap)
    np.testing.assert_allclose(e0, e_diff_swap + e1)


@pytest.mark.parametrize('basis', ['trigonometric'])
def test_swap_binary_cubic(basis):
    seed_rnd_generator(42)
    pri = bulk("H", crystalstructure="sc", a=1.)
    sub = bulk("He", crystalstructure="sc", a=1.)
    plat = ParentLattice(atoms=pri, substitutions=[sub], pbc=(1, 1, 1))
    cpool = ClustersPool(plat, npoints=[1, 2], radii=[0, 2.1])
    mult = cpool.get_multiplicities()

    corcE2 = CorrelationsCalculator("trigonometric", plat, cpool)
    model = Model(corcE2, "energy2",ecis=mult)
    model.reset_mc(True)

    p = [10, 10, 10]
    scell = SuperCell(plat, p)
    structure = scell.gen_random_structure(nsubs=int(np.prod(p)/2))
    i, j = structure.swap_random_binary(site_type=0)

    pred_init = model.predict(structure)
    pred_swap = model.predict_swap(structure, i, j)
    structure.swap(i, j)
    pred_final = model.predict(structure)
    print(pred_init, pred_final)
    print(pred_swap, pred_init - pred_final)


def test_predict_swap_energy_model():
    # binary case
    structure = bulk("Si")
    substitutions = structure.copy()
    substitutions.set_chemical_symbols(["Ge", "Ge"])
    pl = ParentLattice(atoms=structure, substitutions=[substitutions], pbc=(1, 1, 1))
    scell = SuperCell(pl, 2)
    seed_rnd_generator()
    sset = StructuresSet(parent_lattice=pl, calculator=EMT2())
    for nsub in range(2, 15):
        for _ in range(3):
            structure = scell.gen_random_structure(nsubs={0: [nsub]})
            sset.add_structure(structure)
    sset.calculate_property()
    cpool = ClustersPool(pl, npoints=[1, 2], radii=[0, 3])
    mb = ModelBuilder(basis="trigonometric")
    model = mb.build(sset=sset, cpool=cpool, prop="energy")

    # This is a horrible workaround. The behavior of the CorrelationsCalculator is different
    # when performing MC simulations, so we run a short one here to allow for predicting a swap.
    predict_full = MonteCarlo(energy_model=model, scell=scell, nsubs={0: [4]})
    _ = predict_full.metropolis(temperature=100, no_of_sampling_steps=1)

    # To make sure the same indices are swapped in prediction and for the swap
    swap_idx1 = 0
    swap_idx2 = 6

    seed_rnd_generator()
    structure = scell.gen_random_structure(nsubs={0: [5]})
    energy_predict_swap = model.predict_swap(structure, ind1=swap_idx1, ind2=swap_idx2)
    print(
        "Predict swap correlations:",
        model.predict_swap(structure, ind1=swap_idx1, ind2=swap_idx2, correlation=True),
    )
    model.corrc.reset_mc()
    energy_original_predicted = model.predict(structure)
    print(f"Original structure sigmas: {structure.get_sigmas()}")
    print("correlations of original", model.corrc.get_cluster_correlations(structure))
    structure.swap(swap_idx1, swap_idx2)
    print(f"Swapped structure sigmas: {structure.get_sigmas()}")
    energy_swapped_predicted = model.predict(structure)
    print("correlations of swapped", model.corrc.get_cluster_correlations(structure))

    assert not np.isclose(energy_swapped_predicted, energy_original_predicted), (
        "Energies of swapped structure is too similar to energy of original structure"
    )
    swapped_energy = energy_original_predicted - energy_swapped_predicted
    assert np.isclose(swapped_energy, energy_predict_swap), (
        f"Prediction of swap differs from energy difference after swap: real:{swapped_energy}, predicted:{energy_predict_swap}"
    )


def test_predict_swap_monte_carlo_binary():
    structure = bulk("Si")
    substitutions = structure.copy()
    substitutions.set_chemical_symbols(["Ge", "Ge"])
    pl = ParentLattice(atoms=structure, substitutions=[substitutions], pbc=(1, 1, 1))
    scell = SuperCell(pl, 2)
    seed_rnd_generator()
    sset = StructuresSet(parent_lattice=pl, calculator=EMT2())

    for nsub in range(1, 16):
        for _ in range(5):
            structure = scell.gen_random_structure(nsubs={0: [nsub]})
            sset.add_structure(structure)

    sset.calculate_property()

    cpool = ClustersPool(pl, npoints=[1, 2, 3], radii=[0, 3, 4])

    from time import perf_counter

    mb = ModelBuilder()
    mb.initialize()
    model = mb.build(sset=sset, cpool=cpool, prop="energy")

    predict_full = MonteCarlo(energy_model=model, scell=scell, nsubs={0: [4]})
    seed_rnd_generator()
    t1 = perf_counter()
    traj_full = predict_full.metropolis(temperature=100, no_of_sampling_steps=100)
    t2 = perf_counter()
    print(f"time full: {t2 - t1}")
    seed_rnd_generator()
    traj_full2 = predict_full.metropolis(temperature=100, no_of_sampling_steps=100)

    predict_swap = MonteCarlo(
        energy_model=model, scell=scell, nsubs={0: [4]}, predict_swap=True
    )
    seed_rnd_generator()
    t1 = perf_counter()
    traj_swap = predict_swap.metropolis(temperature=100, no_of_sampling_steps=100)
    t2 = perf_counter()
    print(f"time swap: {t2 - t1}")

    assert len(traj_full._trajectory) == len(traj_swap._trajectory), (
        "Different length for trajectories."
    )
    assert np.allclose(traj_full.get_energies(), traj_swap.get_energies()), (
        "Energies of structures visited with 'predict_swap' differ from full CE prediction."
    )
