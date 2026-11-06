"""Local MC optimizations preserve geometry, energy changes, and sampling."""

from copy import deepcopy
import pickle

import numpy as np
import pytest
from ase import Atoms

import clusterx.model as model_module
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator, _cluster_function_flip_indexed, cluster_function_flip
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo
from clusterx.utils import take_tile


def legacy_neighborhood(structure, index):
    """Sequential extraction used before the bounded gather."""
    coordinates = np.unravel_index(index, structure.sigma_grid.shape)
    values, center = take_tile(structure.sigma_grid, coordinates, structure.p_reduced_diag)
    structure.structure_reduced.sigmas = values.reshape(-1)
    return structure.structure_reduced, int(np.ravel_multi_index(center, values.shape))


def legacy_correlations(structure, ind, old_sigma, new_sigma, site_clusters, cluster_orbits_array,
                        cluster_indices, basis_set_values, multiplicities):
    """Reference calculation with the original public per-cluster function."""
    result = np.zeros_like(multiplicities, dtype=float)
    for cluster_id in site_clusters[ind]:
        cluster = cluster_orbits_array[cluster_id]
        sites = cluster.get_idxs()
        result[cluster_indices[cluster_id]] += cluster_function_flip(
            sites, cluster.alphas, structure.sigmas.take(sites), structure.ems.take(sites),
            ind, old_sigma, new_sigma, basis_set_values,
        )
    return np.around(result / multiplicities, decimals=12)


def parent_lattice(pbc=(True, True, False)):
    # An oblique lattice with a ternary sublattice and a spectator site.
    atoms = Atoms("HHe", positions=[[0, 0, 0], [0.3, 0.4, 0.2]],
                  cell=[[1, 0, 0], [0.2, 1.1, 0], [0.1, 0.3, 1.4]], pbc=pbc)
    return ParentLattice(atoms, numbers=[[1, 6, 7], [2]])


@pytest.mark.parametrize("pbc,shape,reduced", [
    ((True, False, False), (4, 1, 1), (3, 1, 1)),
    ((True, True, False), (3, 2, 1), (5, 3, 1)),
    ((True, True, True), (3, 4, 2), (3, 3, 5)),
    ((True, True, False), (2, 3, 1), (2, 2, 1)),
])
def test_neighborhood_order_and_wrapping(pbc, shape, reduced):
    plat = parent_lattice(pbc)
    scell = SuperCell(plat, shape)
    structure = scell.gen_random_structure(rng=np.random.default_rng(23))
    structure.init_reduced_structure(SuperCell(plat, reduced))
    before = structure.sigmas.copy()
    for index in range(len(scell)):
        reference, expected_index = legacy_neighborhood(structure, index)
        expected = reference.sigmas.copy()
        actual, actual_index = structure.get_reduced_structure(np.int64(index))
        np.testing.assert_array_equal(actual.sigmas, expected)
        assert actual_index == expected_index
        assert isinstance(actual_index, int)
        assert actual.sigmas[actual_index] == structure.sigmas[index]
    np.testing.assert_array_equal(structure.sigmas, before)
    assert np.shares_memory(structure.sigmas, structure.sigma_grid)
    assert not np.shares_memory(structure.sigmas, actual.sigmas)


def test_neighborhood_cache_updates_and_old_pickle():
    plat = parent_lattice()
    scell = SuperCell(plat, (4, 4, 1))
    structure = scell.gen_random_structure(rng=np.random.default_rng(29))
    structure.init_reduced_structure(SuperCell(plat, (3, 3, 1)))
    structure.get_reduced_structure(0)
    cache = structure._reduced_tile
    structure.sigmas[0] = (structure.sigmas[0] + 1) % 3
    local, index = structure.get_reduced_structure(0)
    assert structure._reduced_tile is cache
    assert local.sigmas[index] == structure.sigmas[0]
    for old_pickle in (False, True):
        if old_pickle:
            structure.__dict__.pop("_reduced_tile")
            structure.__dict__.pop("_reduced_tile_key")
        restored = pickle.loads(pickle.dumps(structure))
        local, index = restored.get_reduced_structure(0)
        assert local.sigmas[index] == restored.sigmas[0]
        restored.init_reduced_structure(SuperCell(plat, (5, 1, 1)))
        reference, expected_index = legacy_neighborhood(restored, len(scell) - 1)
        expected = reference.sigmas.copy()
        actual, actual_index = restored.get_reduced_structure(len(scell) - 1)
        np.testing.assert_array_equal(actual.sigmas, expected)
        assert actual_index == expected_index
        assert restored._reduced_tile_key[1] == (5, 1, 1)


def test_neighborhood_cache_is_independent_of_system_size():
    plat = parent_lattice()
    reduced = SuperCell(plat, (3, 3, 1))
    for shape in ((3, 3, 1), (12, 12, 1), (48, 1, 1)):
        scell = SuperCell(plat, shape)
        structure = scell.get_pristine_structure()
        structure.init_reduced_structure(reduced)
        structure.get_reduced_structure(0)
        cache = structure._reduced_tile
        for index in (len(scell) // 2, len(scell) - 1):
            local, _ = structure.get_reduced_structure(index)
            assert structure._reduced_tile is cache
            assert local.sigmas.size == len(reduced)
        assert sum(offset.nbytes for offset in cache[:3]) == (3 + 3 + 1) * np.dtype(np.intp).itemsize


def test_neighborhood_invalid_index():
    plat = parent_lattice()
    scell = SuperCell(plat, (3, 3, 1))
    structure = scell.get_pristine_structure()
    with pytest.raises(RuntimeError, match="before structure.init_reduced_structure"):
        structure.get_reduced_structure(0)
    structure.init_reduced_structure(scell)
    for index in (-1, len(scell)):
        with pytest.raises(ValueError, match="out of bounds"):
            structure.get_reduced_structure(index)


@pytest.mark.parametrize("sites", [(5,), (0, 5, 2), (5, 2, 5)])
def test_indexed_cluster_function_preserves_wrapped_site_convention(sites):
    rng = np.random.default_rng(31)
    sigmas = rng.integers(0, 3, size=8, dtype=np.uint8)
    ems = np.full(8, 3, dtype=np.int8)
    alphas = np.arange(len(sites)) % 2 + 1
    basis = rng.normal(size=(3, 3, 4))
    old_sigma = int(sigmas[5])
    new_sigma = (old_sigma + 1) % 3
    expected = cluster_function_flip(sites, alphas, sigmas.take(sites), ems.take(sites),
                                     5, old_sigma, new_sigma, basis)
    actual = _cluster_function_flip_indexed(sites, alphas, sigmas, ems, 5, old_sigma, new_sigma, basis)
    assert actual == expected


@pytest.fixture
def local_model():
    plat = parent_lattice()
    pool = ClustersPool(plat, npoints=[1, 2, 3], radii=[0, 1.55, 1.55])
    calculator = CorrelationsCalculator("trigonometric", plat, pool)
    return Model(calculator, "energy", ecis=np.random.default_rng(37).normal(scale=0.1, size=len(pool)))


@pytest.mark.parametrize("reduce", [False, True])
@pytest.mark.parametrize("basis", ["trigonometric", "polynomial"])
def test_ternary_local_changes_match_full_evaluations(local_model, reduce, basis):
    model = local_model
    model.corrc = CorrelationsCalculator(basis, model.get_parent_lattice(), model.corrc.get_cpool())
    scell = SuperCell(model.get_parent_lattice(), (5, 4, 1))
    rng = np.random.default_rng(41)
    structure = scell.gen_random_structure(rng=rng)
    if reduce:
        model.init_reduced_model(structure)
        structure.init_reduced_structure(model.scell_reduced)
    for index in rng.choice(scell.get_substitutional_sites(), size=12, replace=True):
        old_sigma = int(structure.sigmas[index])
        new_sigma = (old_sigma + 1) % 3
        before = model.predict(structure)
        delta = model.predict_flip(structure, index, old_sigma, new_sigma, reduce_structure=reduce)
        structure.sigmas[index] = new_sigma
        np.testing.assert_allclose(delta, model.predict(structure) - before, atol=2e-12, rtol=1e-10)


def test_fixed_seed_mc_matches_legacy_local_calculation(local_model, monkeypatch, tmp_path):
    scell = SuperCell(local_model.get_parent_lattice(), (5, 4, 1))
    tag = int(scell.get_substitutional_tags()[0])
    nsubs = {tag: [6, 7]}
    initial = scell.gen_random_structure(nsubs=nsubs, rng=np.random.default_rng(43))
    active = scell.get_substitutional_sites()
    counts = np.bincount(initial.sigmas[active], minlength=3)
    results = []
    for legacy in (True, False):
        model = deepcopy(local_model)
        with monkeypatch.context() as context:
            if legacy:
                context.setattr(Structure, "get_reduced_structure", legacy_neighborhood)
                context.setattr(model_module, "cluster_correlations_flip", legacy_correlations)
            mc = MonteCarlo(model, scell, nsubs=nsubs, predict_swap=True, reduce_structure=True, seed=47)
            trajectory = mc.metropolis(
                no_of_sampling_steps=80, temperature=0.3, initial_decoration=initial.get_decoration(),
                serialize=True, filename=str(tmp_path / f"trajectory-{legacy}.json"),
            )
        sigmas = initial.sigmas.copy()
        energy0 = local_model.predict(initial)
        for nid in range(len(trajectory.get_energies())):
            entry = trajectory.get_sampling_step_entry(nid)
            sigmas[entry["indices"]] = entry["new_sigmas"]
            np.testing.assert_array_equal(np.bincount(sigmas[active], minlength=3), counts)
            np.testing.assert_array_equal(sigmas[scell.get_spectator_sites()], 0)
            energy = local_model.predict(Structure(scell, sigmas=sigmas))
            np.testing.assert_allclose(entry["energy"], energy - energy0, atol=2e-12)
        results.append((mc.structure.sigmas.copy(), trajectory.get_sampling_step_nos(), trajectory.get_energies(),
                        deepcopy(mc._rng_np.bit_generator.state), mc._rng_std.getstate()))
    for expected, actual in zip(results[0][:3], results[1][:3]):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_equal(results[0][3], results[1][3])
    assert results[0][4] == results[1][4]
