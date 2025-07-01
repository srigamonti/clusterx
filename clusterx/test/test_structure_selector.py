# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import pytest
from ase import Atoms
import numpy as np

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.structure_selector import StructureSelector


@pytest.fixture
def plat():
    cell = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 5]]
    positions = [[0, 0, 0]]
    pbc = [True, True, False]

    pri = Atoms(['H'], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(['C'], positions=positions, cell=cell, pbc=pbc)

    return ParentLattice(pri, substitutions=[su1], pbc=pbc)


@pytest.fixture
def cpool(plat):
    return ClustersPool(plat, npoints=[1, 2], radii=[0, 1.1])


@pytest.fixture
def scell(plat):
    return SuperCell(plat, np.array([(3, 0, 0), (0, 3, 0), (0, 0, 1)]))


@pytest.fixture
def training_set(plat, scell):
    training_set = StructuresSet(plat)
    n_training = 14
    np.random.seed(42)  # For reproducibility
    for _ in range(n_training):
        training_set.add_structure(scell.gen_random_structure(nsubs={0: [5]}))
    return training_set


@pytest.fixture
def structure_selector(cpool, training_set):
    return StructureSelector(cluster_pool=cpool, training_set=training_set)


def test_covariance_setup(plat, cpool, training_set, structure_selector):
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)
    comat = corrcal.get_correlation_matrix(training_set)
    covariance_matrix_inv = np.dot(comat.T, comat)
    np.testing.assert_allclose(
        np.identity(len(cpool)),
        np.dot(structure_selector._covariance_matrix, covariance_matrix_inv),
        atol=1e-12)


def test_empty_set(structure_selector):
    with pytest.raises(TypeError):
        structure_selector.set_candidate_set(None)


@pytest.mark.parametrize('method', ['global_abc', 'abc'])
def test_invalid_method_select(structure_selector, training_set, method):
    structure_selector.set_candidate_set(training_set)
    with pytest.raises(ValueError):
        structure_selector.select_structure(method=method)


@pytest.mark.parametrize('method', ['global_abc', 'abc'])
def test_invalid_method_domain_matrix(
    structure_selector, training_set, method):
    structure_selector.set_candidate_set(training_set)
    with pytest.raises(ValueError):
        structure_selector._calculate_domain_matrix(method=method)


@pytest.mark.parametrize(
    'method',
    ['averagedConcentration', 'infiniteCrystalFiniteClusters',
     'byConcentration', 'vdWalleAndCeder'])
def test_tau_bounds(structure_selector, method):
    tau = structure_selector.calculate_population_variance(
        domain_calculation_method=method, concentration=0.2)
    assert isinstance(tau, float), "tau is not a float"
    assert tau > 0, "tau is not positive"


@pytest.mark.parametrize('seed', [42])
@pytest.mark.parametrize(
    'method',
    ['global_averagedConcentration', 'global_infiniteCrystalFiniteClusters',
    'global_byConcentration', 'global_vdWalleAndCeder'])
def test_tau_selection(seed, method, plat, scell, cpool, structure_selector, training_set):
    # TODO: test greedy method how?
    calculate_tau_method = method.replace('global_', '')

    candidate_set = StructuresSet(plat)
    n_candidates = 6
    np.random.seed(seed)  # For reproducibility
    for idx in range(n_candidates):
        candidate_set.add_structure(scell.gen_random_structure(nsubs={0:[5]}))
    structure_selector.set_candidate_set(candidate_set)
    structure_idx = structure_selector.select_structure(
        method=method, concentration=0.2)

    tau = np.zeros(n_candidates)
    for candidate_idx in range(n_candidates):
        candidate = candidate_set.get_structure(candidate_idx)
        dummy_set = training_set[:]
        dummy_set.add_structure(candidate)
        dummy_structure_selector = StructureSelector(
            cluster_pool=cpool, training_set=dummy_set)
        tau[candidate_idx] = dummy_structure_selector.calculate_population_variance(
            domain_calculation_method=calculate_tau_method, concentration=0.2)

    structure_chosen = np.argmin(tau)
    assert structure_chosen == structure_idx, f"""structure selection failed. 
        Criteria are not optimally reducing prediction variance according 
        to themselves. generated tau values: {tau}, seed {seed}"""


@pytest.mark.parametrize(
    'method',
    ['averagedConcentration', 'byConcentration',
    'infiniteCrystalFiniteClusters', 'vdWalleAndCeder'])
def test_calc_domain_matrix(method, structure_selector):
    structure_selector._calculate_domain_matrix(method, concentration=0.2)
