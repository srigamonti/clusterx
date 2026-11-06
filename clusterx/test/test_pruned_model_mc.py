"""Pruned builder models retain geometry for reduced-structure Monte Carlo."""

import pickle

import numpy as np
import pytest
from ase import Atoms

from clusterx.cli.build_model_from_sset import build_model_from_sset
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo


@pytest.fixture
def pool():
    # Oblique 2D geometry, a binary sublattice, and spectator atoms.
    atoms = Atoms("HHe", positions=[[0, 0, 0], [0.3, 0.4, 0.2]],
                  cell=[[1, 0, 0], [0.3, 1.1, 0], [0, 0, 4]], pbc=[True, True, False])
    plat = ParentLattice(atoms, numbers=[[1, 7], [2]])
    return ClustersPool(plat, npoints=[1, 2], radii=[0, 1.55])


@pytest.mark.parametrize("selected", [[3, 0], [0], [2, 1]])
def test_subpool_bounds_roundtrip(pool, tmp_path, selected):
    # Subpools retain the original column order, even for unordered selections.
    subpool = pool.get_subpool(selected)
    ordered = sorted(selected)
    assert subpool.get_cpool_list() == [pool[i] for i in ordered]
    np.testing.assert_array_equal(subpool.get_multiplicities(), pool.get_multiplicities()[ordered])
    expected_orders = sorted({pool[i].npoints for i in selected})
    expected_radii = [max(pool[i].get_radius() for i in selected if pool[i].npoints == n) for n in expected_orders]
    np.testing.assert_array_equal(subpool.get_npoints(), expected_orders)
    np.testing.assert_allclose(subpool.get_max_radii(), expected_radii, rtol=1e-15, atol=0)
    filepath = str(tmp_path / "subpool.json")
    subpool.serialize(filepath)
    loaded = ClustersPool(filepath=filepath)
    for candidate in (loaded, pickle.loads(pickle.dumps(subpool))):
        np.testing.assert_array_equal(candidate.get_npoints(), expected_orders)
        np.testing.assert_allclose(candidate.get_max_radii(), expected_radii, rtol=1e-15, atol=0)
        np.testing.assert_array_equal(candidate.get_multiplicities(), subpool.get_multiplicities())
        for actual, expected in zip(candidate.get_cpool_list(), subpool.get_cpool_list()):
            np.testing.assert_allclose(actual.get_positions(), expected.get_positions())
            np.testing.assert_array_equal(actual.get_nrs(), expected.get_nrs())
    pool._radii[1] += 1
    pool._npoints[1] += 1
    np.testing.assert_allclose(subpool.get_max_radii(), expected_radii, rtol=1e-15, atol=0)
    np.testing.assert_array_equal(subpool.get_npoints(), expected_orders)


def test_empty_subpool_has_empty_bounds(pool):
    subpool = pool.get_subpool([])
    assert len(subpool) == 0
    assert subpool.get_npoints().size == 0
    assert subpool.get_max_radii().size == 0


def test_subpool_computes_a_separate_bound_for_each_order(pool):
    pool = ClustersPool(pool.get_plat(), npoints=[1, 2, 3], radii=[0, 1.55, 1.55])
    pairs = [i for i, cl in enumerate(pool.get_cpool_list()) if cl.npoints == 2]
    triplets = [i for i, cl in enumerate(pool.get_cpool_list()) if cl.npoints == 3]
    assert len(triplets) > 0
    subpool = pool.get_subpool([pairs[0], *triplets])
    np.testing.assert_array_equal(subpool.get_npoints(), [2, 3])
    expected = [pool[pairs[0]].get_radius(), max(pool[i].get_radius() for i in triplets)]
    np.testing.assert_allclose(subpool.get_max_radii(), expected)


def test_subpool_bound_does_not_wrap_cluster_positions(pool):
    # A pair spanning three cells must not be shortened by a minimum-image calculation.
    pair = pool.get_cpool_list()[-1]
    pair.positions_cartesian = np.array([[0, 0, 0], [3, 0, 0]], dtype=float)
    subpool = pool.get_subpool([len(pool) - 1])
    np.testing.assert_allclose(subpool.get_max_radii(), [3])


@pytest.mark.parametrize("singlets_only", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_builder_pruned_model_reduced_mc(pool, tmp_path, monkeypatch, singlets_only, standardize):
    plat = pool.get_plat()
    training_cell = SuperCell(plat, [3, 3, 1])
    training = StructuresSet(plat)
    rng = np.random.default_rng(53)
    for _ in range(48):
        training.add_structure(training_cell.gen_random_structure(rng=rng))
    calculator = CorrelationsCalculator("trigonometric", plat, pool)
    correlations = calculator.get_correlation_matrix(training)
    targets = 1.25 + 0.7 * correlations[:, 0]
    if not singlets_only:
        targets += 1.7 * correlations[:, 1]
    training.set_property_values("prop", targets)
    source = str(tmp_path / "training.xyz")
    training.serialize(source)
    prefix = str(tmp_path / "fit")
    linear = {"module": "sklearn.linear_model", "class": "LinearRegression", "args": {}}
    build_model_from_sset(
        "prop", source, npoints=[1, 2], radii=[0, 1.55], output_prefix=prefix,
        regression_model=linear, selection_model=linear, selection_threshold=1e-8,
        nonlinear_transformation={"args": {"degree": 1, "include_bias": False}},
        standardize=standardize,
    )
    full = Model(filepath=f"{prefix}_1_full_model.pickle")
    pruned = Model(filepath=f"{prefix}_2_pruned_model.pickle")
    pruned_pool = pruned.corrc.get_cpool()
    assert full._get_affine_parameters(len(full.corrc.get_cpool())) is not None
    assert pruned._get_affine_parameters(len(pruned_pool)) is not None
    assert 0 < len(pruned_pool) < len(full.corrc.get_cpool())
    np.testing.assert_allclose([full.predict(s) for s in training], targets, atol=1e-11)
    np.testing.assert_allclose([pruned.predict(s) for s in training], targets, atol=1e-11)
    if singlets_only:
        assert (pruned_pool.get_all_npoints() == 1).all()
    else:
        assert (pruned_pool.get_all_npoints() > 1).any()
    scell = SuperCell(plat, [7, 6, 1])
    active = scell.get_substitutional_sites()
    tag = int(plat.get_tags()[0])
    nsubs = {tag: [len(active) // 2]}
    initial = scell.gen_random_structure(nsubs=nsubs, rng=np.random.default_rng(37))
    original_flip = pruned.predict_flip
    checked = []

    def checked_flip(structure, index, old_sigma, new_sigma, reduce_structure=False):
        assert reduce_structure
        before = structure.sigmas.copy()
        delta = original_flip(structure, index, old_sigma, new_sigma, reduce_structure)
        changed = Structure(scell, sigmas=before.copy())
        changed.sigmas[index] = new_sigma
        expected = full.predict(changed) - full.predict(structure)
        np.testing.assert_allclose(delta, expected, atol=2e-12, rtol=1e-10)
        np.testing.assert_array_equal(structure.sigmas, before)
        checked.append(delta)
        return delta

    monkeypatch.setattr(pruned, "predict_flip", checked_flip)
    mc = MonteCarlo(pruned, scell, nsubs=nsubs, predict_swap=True, reduce_structure=True, seed=37)
    trajectory = mc.metropolis(
        no_of_sampling_steps=24, temperature=0.5, boltzmann_constant=1.0, scale_factor=[1.0],
        initial_decoration=initial.get_atoms().numbers, serialize=True,
    )
    assert len(checked) == 48
    assert len(trajectory.get_energies()) > 1
    np.testing.assert_array_equal(pruned.p_reduced.diagonal()[2:], [1])
    if singlets_only:
        np.testing.assert_array_equal(pruned.p_reduced, np.eye(3, dtype=int))
    else:
        assert np.any(np.abs(trajectory.get_energies()) > 1e-8)

    sigmas = initial.sigmas.copy()
    counts = np.bincount(sigmas[active], minlength=2)
    energy0 = full.predict(initial)
    for nid in range(len(trajectory.get_energies())):
        entry = trajectory.get_sampling_step_entry(nid)
        sigmas[entry["indices"]] = entry["new_sigmas"]
        np.testing.assert_array_equal(np.bincount(sigmas[active], minlength=2), counts)
        np.testing.assert_array_equal(sigmas[scell.get_spectator_sites()], 0)
        structure = Structure(scell, sigmas=sigmas.copy())
        energy = full.predict(structure)
        np.testing.assert_allclose(pruned.predict(structure), energy, atol=1e-11)
        np.testing.assert_allclose(entry["energy"], energy - energy0, atol=1e-11)
