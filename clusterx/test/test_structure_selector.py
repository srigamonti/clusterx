# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import pytest
import scipy
from ase import Atoms
import numpy as np

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.structure_selector import StructureSelector
from clusterx.structure import Structure


@pytest.fixture
def plat():
    cell = [[1, 0, 0], [0, 1, 0], [0, 0, 5]]
    positions = [[0, 0, 0]]
    pbc = [True, True, False]

    pri = Atoms(["H"], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(["C"], positions=positions, cell=cell, pbc=pbc)

    return ParentLattice(pri, substitutions=[su1], pbc=pbc)


@pytest.fixture
def cpool(plat):
    return ClustersPool(plat, npoints=[1, 2], radii=[0, 1.1])

@pytest.fixture
def cpool_candidates(plat):
    return ClustersPool(plat, npoints=[1, 2, 3], radii=[0, 1.1, 2.2])

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
def candidate_set(plat, scell):
    candidate_set = StructuresSet(plat)
    n_candidates = 6
    np.random.seed(42)  # For reproducibility
    for idx in range(n_candidates):
        candidate_set.add_structure(scell.gen_random_structure(nsubs={0: [5]}))
    return candidate_set


@pytest.fixture
def structure_selector(cpool, training_set, candidate_set):
    return StructureSelector(cpool=cpool, training_set=training_set, candidate_set=candidate_set)


def test_covariance_setup(plat, cpool, training_set, structure_selector):
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)
    comat = corrcal.get_correlation_matrix(training_set)
    covariance_matrix_inv = np.dot(comat.T, comat)
    np.testing.assert_allclose(
        np.identity(len(cpool)),
        np.dot(structure_selector._covariance_matrix, covariance_matrix_inv),
        atol=1e-12,
    )


def test_empty_set(structure_selector):
    with pytest.raises(TypeError):
        structure_selector.set_candidate_set(None)


@pytest.mark.parametrize("method", ["global_abc", "abc"])
def test_invalid_method_select(structure_selector, method):
    with pytest.raises(ValueError):
        structure_selector.select_structure(method=method)


@pytest.mark.parametrize("method", ["global_abc", "abc"])
def test_invalid_method_domain_matrix(structure_selector, method):
    with pytest.raises(ValueError):
        structure_selector._calculate_domain_matrix(method=method)


@pytest.mark.parametrize(
    "method",
    [
        "averagedConcentration",
        "infiniteCrystalFiniteClusters",
        "byConcentration",
        "vdWalleAndCeder",
    ],
)
def test_tau_bounds(structure_selector, method):
    tau = structure_selector.calculate_population_variance(
        domain_calculation_method=method, concentration=0.2
    )
    assert isinstance(tau, float), "tau is not a float"
    assert tau > 0, "tau is not positive"


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "method",
    [
        "global_averagedConcentration",
        "global_infiniteCrystalFiniteClusters",
        "global_byConcentration",
        "global_vdWalleAndCeder",
    ],
)
def test_structure_selection(
    seed, method, plat, scell, cpool, structure_selector, training_set, candidate_set
):
    # TODO: test greedy method how?
    calculate_tau_method = method.replace("global_", "")
    n_candidates = len(candidate_set)
    structure_idx = structure_selector.select_structure(
        method=method, concentration=0.2
    )

    tau = np.zeros(n_candidates)
    for candidate_idx in range(n_candidates):
        candidate = candidate_set.get_structure(candidate_idx)
        dummy_set = training_set[:]
        dummy_set.add_structure(candidate)
        dummy_structure_selector = StructureSelector(
            cpool=cpool, training_set=dummy_set
        )
        tau[candidate_idx] = dummy_structure_selector.calculate_population_variance(
            domain_calculation_method=calculate_tau_method, concentration=0.2
        )

    structure_chosen = np.argmin(tau)
    assert structure_chosen == structure_idx, f"""structure selection failed.
        Criteria are not optimally reducing prediction variance according
        to themselves. generated tau values: {tau}, seed {seed}"""


@pytest.mark.parametrize(
    "method",
    [
        "averagedConcentration",
        "byConcentration",
        "infiniteCrystalFiniteClusters",
        "vdWalleAndCeder",
    ],
)
def test_calc_domain_matrix(method, structure_selector):
    structure_selector._calculate_domain_matrix(method, concentration=0.2)

def test_integral_calculation_averagedConcentration(structure_selector):
    lower_conc_limit=0.1
    upper_conc_limit=0.5
    domain_matrix = structure_selector._calculate_domain_matrix(method='averagedConcentration',
                                                                lower_conc_limit=lower_conc_limit,
                                                                upper_conc_limit=upper_conc_limit)
    npoints = structure_selector._cpool.get_all_npoints()
    number_clusters = len(npoints)
    domain_matrix_control = np.zeros((number_clusters, number_clusters), dtype=float)
    for rowIdx in range(number_clusters):
        for colIdx in range(number_clusters):
            sum_points = npoints[rowIdx] + npoints[colIdx]
            domain_matrix_control[rowIdx, colIdx] = scipy.integrate.quad(lambda c: np.power((2*c-1), sum_points),
                                                                            lower_conc_limit, upper_conc_limit)[0]
    np.testing.assert_allclose(
            domain_matrix,
            domain_matrix_control,
            atol=1e-12,
            err_msg="Domain matrix is not computed correctly for averagedConcentration method. Check the integration."
        )

# note: test may not be neccessary; delete?
# tested beforehand for which concentration limits another structure is selected then for limits 0 to 1
# therefore test cannot be reproduced with random integral limits, as it could be that same structures selected then
def test_structure_selection_with_differing_concentration_limits(structure_selector):
    idx1 = structure_selector.select_structure(method="global_averagedConcentration",
                                                        lower_conc_limit=0, upper_conc_limit=1)
    idx2 = structure_selector.select_structure(method="global_averagedConcentration",
                                                        lower_conc_limit=0, upper_conc_limit=0.05)
    assert idx1 != idx2, "Same structure selected, even though settings should lead to different selected structures"

@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "method",
    [
        "global_averagedConcentration",
        "global_infiniteCrystalFiniteClusters",
        "global_byConcentration",
        "global_vdWalleAndCeder",
    ],
)
def test_structure_object(
            seed, method, plat, scell,
            candidate_set,
            structure_selector
):
    selected_structure_idx = structure_selector.select_structure(
        method=method, concentration=0.2)
    structure = candidate_set.get_structure(selected_structure_idx)
    assert isinstance(structure, Structure),"selected structure is not a Structure object"

@pytest.mark.parametrize(
    "method",
    [
        "global_averagedConcentration",
        "global_infiniteCrystalFiniteClusters",
        "global_byConcentration",
        "global_vdWalleAndCeder",
    ],
)
def test_input_validation_for_select_structures_set(method, structure_selector):
    # note: method input already tested above and same code was used, so its not tested again here
    if method == "global_byConcentration":
        with pytest.raises(ValueError, match="To select with global_byConcentration"
                                            "please provide a list of concentrations to select for."):
            structure_selector.select_structures_set(
                                                    concentrations=None,
                                                    method=method)
        # note: may be deleted if concentration range is incorrect
        with pytest.raises(ValueError, match="Concentrations need to be between 0 and 1."):
            structure_selector.select_structures_set(concentrations=[1.1, -0.5],
                                                    method=method)
    else:
        with pytest.raises(ValueError, match="Please provide the number of structures you want to select."):
             structure_selector.select_structures_set(
                                                    nr_structures=None,
                                                    method=method)

def test_reduction_candidate_set(structure_selector, candidate_set, cpool_candidates):
    method = "global_averagedConcentration" # note: any method could be chosen here
    structure_selector.select_structures_set(nr_structures=1,
                                            method=method)
    len_candidate_set_wo_reduction = len(structure_selector._candidate_set)
    structure_selector.set_candidate_set(candidate_set)
    structure_selector.select_structures_set(exclude_equivalent_candidates=True,
                                            cpool_candidates=cpool_candidates,
                                            nr_structures=1,
                                            method=method)
    len_candidate_set_w_reduction = len(structure_selector._candidate_set)
    assert len_candidate_set_w_reduction < len_candidate_set_wo_reduction, ("Candidate set was not reduced.")

@pytest.mark.parametrize(
    "method",
    [
        "global_averagedConcentration",
        "global_infiniteCrystalFiniteClusters",
        "global_byConcentration",
        "global_vdWalleAndCeder",
    ],
)
def test_warning_reduction_candidate_set(plat, cpool_candidates, candidate_set, structure_selector, method):
    basis = structure_selector._correlations_calculator.get_basis()
    correlations_calcluator_candidates = CorrelationsCalculator(basis, plat, cpool_candidates)
    candidate_correlations = correlations_calcluator_candidates.get_correlation_matrix(candidate_set)
    print(f"Candidate correlations {candidate_correlations}")
    nr_rows = candidate_correlations.shape[0]
    nr_rows_to_replace = int(0.4*nr_rows)
    print(f"Rows replaced in correlation matrix {nr_rows_to_replace}")
    # choose random columns to replace
    row_idxs_to_replace = np.random.choice(nr_rows, nr_rows_to_replace, replace=False)
    candidate_correlations[row_idxs_to_replace, :] = candidate_correlations[0, :]
    if method == "global_byConcentration":
        with pytest.warns(UserWarning, match="Reduced candidate set contains less than 80 % of original candidate set."
                "Consider increasing cpool_candidates to reduce risk of excluding symmetrically distinct structures."):
            structure_selector.select_structures_set(exclude_equivalent_candidates=True,
                                                    candidate_correlations=candidate_correlations,
                                                    concentrations=[0.1, 0.2],
                                                    method=method)
    else:
        with pytest.warns(UserWarning, match="Reduced candidate set contains less than 80 % of original candidate set."
                "Consider increasing cpool_candidates to reduce risk of excluding symmetrically distinct structures."):
            structure_selector.select_structures_set(exclude_equivalent_candidates=True,
                                                    candidate_correlations=candidate_correlations,
                                                    nr_structures=2,
                                                    method=method)

@pytest.mark.parametrize(
    "method",
    [
        "global_averagedConcentration",
        "global_infiniteCrystalFiniteClusters",
        "global_byConcentration",
        "global_vdWalleAndCeder",
    ],
)
def test_structure_set_selection(structure_selector, method, candidate_set):
    len_candidate_set_it0 = len(structure_selector._candidate_set)
    len_training_set_it0 = len(structure_selector._training_set)
    idx0 = structure_selector.select_structure(method=method, concentration=0.2)
    structure0 = structure_selector._candidate_set.get_structure(idx0)
    sset0 = structure_selector.select_structures_set(method=method,
                                                            nr_structures=1,
                                                            concentrations=[0.2])
    structure0_control = sset0.get_structure(0)
    len_candidate_set_it1 = len(structure_selector._candidate_set)
    len_training_set_it1 = len(structure_selector._training_set)
    assert len_candidate_set_it1 == len_candidate_set_it0 - 1, "Candidate set not reduced as expected"
    assert len_training_set_it1 == len_training_set_it0 + 1, "Training set not extended as expected"
    idx1 = structure_selector.select_structure(method=method, concentration=0.3)
    structure1 = structure_selector._candidate_set.get_structure(idx1)
    sset1 = structure_selector.select_structures_set(method=method,
                                                    nr_structures=1,
                                                    concentrations=[0.3])
    structure1_control = sset1.get_structure(0)
    sset = sset0 + sset1
    assert structure0 == structure0_control, "Not the same structures selected in first iteration"
    assert structure1 == structure1_control, "Not the same structures selected in second iteration"
    structure_selector.set_candidate_set(candidate_set)
    sset_control = structure_selector.select_structures_set(method=method,
                                                            nr_structures=2,
                                                            concentrations=[0.2, 0.3])
    assert sset.get_structure(0) == sset_control.get_structure(0), "Selected structures set does not match control set"
    assert sset.get_structure(1) == sset_control.get_structure(1), "Selected structures set does not match control set"
