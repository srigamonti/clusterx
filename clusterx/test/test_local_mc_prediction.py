"""Local MC optimizations preserve geometry, energy changes, and sampling."""

from copy import deepcopy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

import clusterx.model as model_module
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import (
    CorrelationsCalculator,
    _cluster_correlations_flip_packed,
    _cluster_function_flip_indexed,
    _pack_cluster_interactions,
    cluster_function_flip,
)
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
    cluster = SimpleNamespace(get_idxs=lambda: sites, alphas=alphas)
    orbits = np.array([cluster], dtype=object)
    site_clusters = [[] for _ in sigmas]
    for site in sites:
        site_clusters[site].append(0)
    args = dict(structure=SimpleNamespace(sigmas=sigmas, ems=ems), ind=5,
                old_sigma=old_sigma, new_sigma=new_sigma, cluster_indices=np.array([0]),
                basis_set_values=basis, multiplicities=np.array([1]))
    expected = legacy_correlations(**args, site_clusters=site_clusters, cluster_orbits_array=orbits)
    packed = _pack_cluster_interactions(site_clusters, orbits)
    np.testing.assert_array_equal(_cluster_correlations_flip_packed(**args, packed_interactions=packed), expected)


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
    for reference in ("legacy", "direct", "packed"):
        model = deepcopy(local_model)
        with monkeypatch.context() as context:
            if reference != "packed":
                context.setattr(model_module, "_pack_cluster_interactions", lambda *args: None)
            if reference == "legacy":
                context.setattr(Structure, "get_reduced_structure", legacy_neighborhood)
                context.setattr(model_module, "cluster_correlations_flip", legacy_correlations)
            mc = MonteCarlo(model, scell, nsubs=nsubs, predict_swap=True, reduce_structure=True, seed=47)
            trajectory = mc.metropolis(
                no_of_sampling_steps=80, temperature=0.3, initial_decoration=initial.get_decoration(),
                serialize=True, filename=str(tmp_path / f"trajectory-{reference}.json"),
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
    for result in results[1:]:
        for expected, actual in zip(results[0][:3], result[:3]):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_equal(results[0][3], result[3])
        assert results[0][4] == result[4]


def test_packed_cache_is_independent_of_full_system_size(local_model):
    model = local_model
    plat = model.get_parent_lattice()
    first_cache = None
    for shape in ((5, 4, 1), (10, 8, 1), (40, 1, 1)):
        scell = SuperCell(plat, shape)
        structure = scell.gen_random_structure(rng=np.random.default_rng(59))
        model.init_reduced_model(structure)
        structure.init_reduced_structure(model.scell_reduced)
        site = scell.get_substitutional_sites()[0]
        sigma = int(structure.sigmas[site])
        model.predict_flip(structure, site, sigma, (sigma + 1) % 3, reduce_structure=True)
        cache = model._packed_interactions
        assert cache is not None
        if first_cache is None:
            first_cache = cache
        assert cache is first_cache
        assert cache[0].size == len(model.scell_reduced) + 1
        assert sum(a.nbytes for a in cache) <= 16 * 1024**2


def test_packed_cache_rebuilds_when_reduced_geometry_changes(local_model):
    model = local_model
    plat = model.get_parent_lattice()
    structure = SuperCell(plat, (8, 8, 1)).gen_random_structure(rng=np.random.default_rng(61))
    index = int(structure.get_substitutional_sites()[0])
    old_sigma = int(structure.sigmas[index])
    new_sigma = (old_sigma + 1) % 3
    changed = Structure(structure.get_supercell(), sigmas=structure.sigmas)
    changed.sigmas[index] = new_sigma
    expected = model.predict(changed) - model.predict(structure)
    previous = None
    # Equal site counts but different transformations must invalidate the cache.
    for shape in ((5, 7, 1), (7, 5, 1)):
        reduced = SuperCell(plat, shape)
        structure.init_reduced_structure(reduced)
        model.scell_factor = len(structure.sigmas) / len(reduced)
        actual = model.predict_flip(structure, index, old_sigma, new_sigma, reduce_structure=True)
        np.testing.assert_allclose(actual, expected, atol=2e-12)
        assert model._packed_interactions is not previous
        previous = model._packed_interactions
    model._init_interaction_dict(reduced)
    assert "_packed_interactions" not in model.__dict__
    model.predict_flip(structure, index, old_sigma, new_sigma, reduce_structure=True)
    assert model._packed_interactions is not previous
    model.reset_mc()
    assert "_packed_interactions" not in model.__dict__


def test_unreduced_predictions_do_not_pack_full_geometry(local_model, monkeypatch):
    def unexpected_pack(*args):
        pytest.fail("Unreduced predictions must not allocate packed geometry")

    monkeypatch.setattr(model_module, "_pack_cluster_interactions", unexpected_pack)
    model = local_model
    scell = SuperCell(model.get_parent_lattice(), (5, 4, 1))
    structure = scell.gen_random_structure(rng=np.random.default_rng(67))
    index = scell.get_substitutional_sites()[0]
    sigma = int(structure.sigmas[index])
    model.predict_flip(structure, index, sigma, (sigma + 1) % 3)
    assert "_packed_interactions" not in model.__dict__


def test_packed_budget_declines_before_allocation_and_falls_back(local_model, monkeypatch):
    model = local_model
    scell = SuperCell(model.get_parent_lattice(), (5, 4, 1))
    structure = scell.gen_random_structure(rng=np.random.default_rng(71))
    model.init_reduced_model(structure)
    model._init_interaction_dict(model.scell_reduced)
    structure.init_reduced_structure(model.scell_reduced)
    index = scell.get_substitutional_sites()[0]
    sigma = int(structure.sigmas[index])

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("An oversized packed cache must be declined before allocation")

    with monkeypatch.context() as context:
        context.setattr(np, "empty", unexpected_allocation)
        assert _pack_cluster_interactions(model._site_clusters, model._cluster_orbits_array, max_bytes=1) is None
    calls = []

    def limited_pack(*args):
        calls.append(1)
        return _pack_cluster_interactions(*args, max_bytes=1)

    monkeypatch.setattr(model_module, "_pack_cluster_interactions", limited_pack)
    changed = Structure(scell, sigmas=structure.sigmas)
    changed.sigmas[index] = (sigma + 1) % 3
    expected = model.predict(changed) - model.predict(structure)
    for _ in range(2):
        actual = model.predict_flip(structure, index, sigma, (sigma + 1) % 3, reduce_structure=True)
        np.testing.assert_allclose(actual, expected, atol=2e-12)
        assert model._packed_interactions is None
    assert len(calls) == 1


def test_packed_cache_serialization_and_old_state(local_model, tmp_path):
    model = local_model
    scell = SuperCell(model.get_parent_lattice(), (5, 4, 1))
    structure = scell.gen_random_structure(rng=np.random.default_rng(73))
    model.init_reduced_model(structure)
    structure.init_reduced_structure(model.scell_reduced)
    index = scell.get_substitutional_sites()[0]
    sigma = int(structure.sigmas[index])
    expected = model.predict_flip(structure, index, sigma, (sigma + 1) % 3, reduce_structure=True)
    filepath = str(tmp_path / "packed-model.pickle")
    model.serialize(filepath)
    state = model.__getstate__()
    assert "_packed_interactions" not in state
    assert "_interaction_scell_key" not in state
    old = Model.__new__(Model)
    old.__setstate__(state)
    for restored in (Model(filepath=filepath), pickle.loads(pickle.dumps(model)), old):
        assert "_packed_interactions" not in restored.__dict__
        actual = restored.predict_flip(structure, index, sigma, (sigma + 1) % 3, reduce_structure=True)
        assert restored._packed_interactions is not None
        assert restored._packed_interactions is not model._packed_interactions
        np.testing.assert_array_equal(actual, expected)
