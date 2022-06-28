from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet
from clusterx.model import ModelBuilder
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.calculators.emt import EMT2

from clusterx.thermodynamics.monte_carlo import MonteCarlo

from ase.build import bulk

import numpy as np
import random

def seed_rnd_generator(seed = 0):
    random.seed(seed)
    np.random.seed(seed)

# binary case

def test_predict_swap_energy_model():
    structure = bulk("Si")
    substitutions = structure.copy()
    substitutions.set_chemical_symbols(["Ge", "Ge"])
    pl = ParentLattice(atoms=structure, substitutions=[substitutions], pbc=(1,1,1))
    scell = SuperCell(pl, 2)
    seed_rnd_generator()
    sset = StructuresSet(parent_lattice=pl, calculator=EMT2())
    for nsub in range(2,15):
        for _ in range(3):
            structure = scell.gen_random_structure(nsubs={0:[nsub]})
            sset.add_structure(structure)
    sset.calculate_property()
    cpool = ClustersPool(pl, npoints=[1,2], radii=[0,3])
    mb = ModelBuilder(basis = "trigonometric")
    model = mb.build(sset=sset, cpool=cpool, prop="energy")

    # This is a horrible workaround. The behavior of the CorrelationsCalculator is different 
    # when performing MC simulations, so we run a short one here to allow for predicting a swap.
    predict_full = MonteCarlo(energy_model=model, scell = scell, nsubs={0:[4]})
    _ = predict_full.metropolis(temperature=100, no_of_sampling_steps = 1)

    # To make sure the same indices are swapped in prediction and for the swap
    swap_idx1 = 0
    swap_idx2 = 6

    seed_rnd_generator()
    structure = scell.gen_random_structure(nsubs={0:[5]})
    energy_predict_swap = model.predict_swap(structure, ind1=swap_idx1, ind2=swap_idx2)
    print("Predict swap correlations:", model.predict_swap(structure, ind1=swap_idx1, ind2=swap_idx2, correlation=True))
    model.corrc.reset_mc()
    energy_original_predicted = model.predict(structure)
    print(f"Original structure sigmas: {structure.get_sigmas()}")
    print("correlations of original", model.corrc.get_cluster_correlations(structure))
    structure.swap(swap_idx1,swap_idx2)
    print(f"Swapped structure sigmas: {structure.get_sigmas()}")
    energy_swapped_predicted = model.predict(structure)
    print("correlations of swapped", model.corrc.get_cluster_correlations(structure))

    assert not np.isclose(energy_swapped_predicted, energy_original_predicted), "Energies of swapped structure is too similar to energy of original structure"
    swapped_energy = energy_original_predicted - energy_swapped_predicted
    assert np.isclose(swapped_energy, energy_predict_swap), f"Prediction of swap differs from energy difference after swap: real:{swapped_energy}, predicted:{energy_predict_swap}"

def test_predict_swap_monte_carlo_binary():
    structure = bulk("Si")
    substitutions = structure.copy()
    substitutions.set_chemical_symbols(["Ge", "Ge"])
    pl = ParentLattice(atoms=structure, substitutions=[substitutions], pbc=(1,1,1))
    scell = SuperCell(pl, 2)
    seed_rnd_generator()
    sset = StructuresSet(parent_lattice=pl, calculator=EMT2())

    for nsub in range(1,16):
        for _ in range(5):
            structure = scell.gen_random_structure(nsubs={0:[nsub]})
            sset.add_structure(structure)

    sset.calculate_property()

    cpool = ClustersPool(pl, npoints=[1,2,3], radii=[0,3,4])

    from time import perf_counter

    mb = ModelBuilder()
    mb.initialize()
    model = mb.build(sset=sset, cpool=cpool, prop="energy")

    predict_full = MonteCarlo(energy_model=model, scell = scell, nsubs={0:[4]})
    seed_rnd_generator()
    t1 = perf_counter()
    traj_full = predict_full.metropolis(temperature=100, no_of_sampling_steps = 100)
    t2 = perf_counter()
    print(f"time full: {t2-t1}")
    seed_rnd_generator()
    traj_full2 = predict_full.metropolis(temperature=100, no_of_sampling_steps = 100)

    predict_swap = MonteCarlo(energy_model=model, scell = scell, nsubs={0:[4]}, predict_swap=True)
    seed_rnd_generator()
    t1 = perf_counter()
    traj_swap = predict_swap.metropolis(temperature=100, no_of_sampling_steps = 100)
    t2 = perf_counter()
    print(f"time swap: {t2-t1}")

    assert len(traj_full._trajectory) == len(traj_swap._trajectory), "Different length for trajectories."
    assert np.allclose(traj_full.get_energies(), traj_swap.get_energies()), "Energies of structures visited with 'predict_swap' differ from full CE prediction."