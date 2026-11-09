"""Tests for memory-bounded and resumable Monte Carlo runs."""

import random

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo


def make_system():
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=[True, False, False])
    plat = ParentLattice(atoms, substitutions=[Atoms("N", positions=[[0, 0, 0]], cell=[1, 1, 1],
                                                        pbc=[True, False, False])])
    pool = ClustersPool(plat, npoints=[1, 2], radii=[0, 1.01])
    model = Model(CorrelationsCalculator("trigonometric", plat, pool), "energy", ecis=np.array([0.2, -0.4]))
    return plat, SuperCell(plat, (16, 1, 1)), model


def test_batched_proposals_match_uninterrupted_and_bound_beta_arrays(monkeypatch):
    _, scell, model = make_system()
    expected_mc = MonteCarlo(model, scell, nsubs={0: [8]}, seed=102)
    expected_mc.metropolis(120, temperature=0.8)
    expected_sigmas = expected_mc.structure.get_sigmas().copy()
    expected_np_state = expected_mc._rng_np.bit_generator.state
    expected_py_state = random.getstate()

    sizes = []
    original = MonteCarlo._iter_beta_chunks

    def tracked(*args):
        for start, values in original(*args):
            sizes.append(len(values))
            yield start, values

    monkeypatch.setattr(MonteCarlo, "_iter_beta_chunks", staticmethod(tracked))
    actual_mc = MonteCarlo(model, scell, nsubs={0: [8]}, seed=102)
    actual_mc.metropolis(120, temperature=0.8, max_steps_per_batch=7)
    np.testing.assert_array_equal(actual_mc.structure.get_sigmas(), expected_sigmas)
    actual_np_state = actual_mc._rng_np.bit_generator.state
    np.testing.assert_array_equal(actual_np_state["state"]["state"], expected_np_state["state"]["state"])
    assert random.getstate() == expected_py_state
    assert max(sizes) <= 7
    assert len(sizes) == 18
    assert actual_mc.structure.get_fractional_concentrations()[0][0] == 0.5


def test_xyz_resume_uses_last_frame_and_appends_checkpoints(tmp_path):
    _, scell, model = make_system()
    first = Structure(scell, sigmas=np.array([0, 0] * 8), track_indices=True)
    second_sigmas = np.array([0, 1] * 8)
    second = Structure(scell, sigmas=second_sigmas, track_indices=True)
    path = tmp_path / "temperatures.extxyz"
    frame1 = first.get_atoms()
    frame1.info["temperature"] = 900
    frame2 = second.get_atoms()
    frame2.info["temperature"] = 500
    write(path, [frame1, frame2], format="extxyz")

    mc = MonteCarlo(model, scell, nsubs={0: [8]}, seed=204, resume_from=str(path),
                    save_atoms_every=1, atoms_filename=str(path))
    trajectory = mc.metropolis(5, temperature=0.1, serialize=True)
    np.testing.assert_array_equal(trajectory._trajectory[0]["sigmas"], second_sigmas)
    assert len(mc.structure) == len(scell)
    assert mc.structure.get_fractional_concentrations()[0][0] == 0.5

    from ase.io import read

    frames = read(path, index=":", format="extxyz")
    # Resume does not duplicate the last source frame; checkpoints are appended.
    assert len(frames) == 7
    np.testing.assert_array_equal(frames[0].arrays["sigmas"], first.get_sigmas())
    np.testing.assert_array_equal(frames[1].arrays["sigmas"], second_sigmas)
    np.testing.assert_array_equal(frames[-1].arrays["sigmas"], mc.structure.get_sigmas())


def test_resume_rejects_a_frame_with_the_wrong_canonical_composition(tmp_path):
    _, scell, model = make_system()
    structure = Structure(scell, sigmas=np.array([0, 1] * 8), track_indices=True)
    path = tmp_path / "wrong-composition.xyz"
    write(path, structure.get_atoms(), format="extxyz")
    mc = MonteCarlo(model, scell, nsubs={0: [7]}, resume_from=str(path), seed=3)
    with pytest.raises(ValueError, match="composition"):
        mc.metropolis(1)


def test_reset_chain_reuses_requested_initial_decoration():
    _, scell, model = make_system()
    mc = MonteCarlo(model, scell, nsubs={0: [8]}, seed=1)
    expected = np.array([0, 1] * 8)
    # Atomic numbers in this lattice are H=1 and N=7; use an explicit decoration.
    mc.reset_chain(initial_decoration=[7 if sigma else 1 for sigma in expected], seed=77)
    mc.metropolis(0, continue_chain=True)
    np.testing.assert_array_equal(mc.structure.get_sigmas(), expected)


def test_nonlinear_continuation_keeps_correlation_cache(monkeypatch):
    _, scell, model = make_system()
    width = len(model.corrc.get_cpool())
    features = np.random.default_rng(12).normal(size=(100, width))
    targets = features[:, 0] ** 2 + features[:, -1] * 0.3
    model.estimator = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(features, targets)

    calls = 0
    original = model.get_correlations

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "get_correlations", counted)
    whole = MonteCarlo(
        model, scell, nsubs={0: [8]}, predict_swap=True, reduce_structure=True, seed=305
    )
    whole.metropolis(20, temperature=0.4, nonlinear=True, max_steps_per_batch=3)
    whole_sigmas = whole.structure.get_sigmas().copy()
    whole_rng = whole._rng_np.bit_generator.state
    assert calls == 1

    calls = 0
    split = MonteCarlo(
        model, scell, nsubs={0: [8]}, predict_swap=True, reduce_structure=True, seed=305
    )
    split.metropolis(8, temperature=0.4, nonlinear=True, max_steps_per_batch=3)
    split.metropolis(12, temperature=0.4, nonlinear=True, continue_chain=True, max_steps_per_batch=5)
    np.testing.assert_array_equal(split.structure.get_sigmas(), whole_sigmas)
    split_rng = split._rng_np.bit_generator.state
    np.testing.assert_array_equal(split_rng["state"]["state"], whole_rng["state"]["state"])
    assert calls == 1


def test_target_rate_adaptation_continues_across_calls():
    _, scell, model = make_system()
    whole = MonteCarlo(model, scell, nsubs={0: [8]}, seed=409)
    whole.metropolis(1_100, temperature=0.5, target_rate=0.35, max_steps_per_batch=113)
    expected_sigmas = whole.structure.get_sigmas().copy()
    expected_rng = whole._rng_np.bit_generator.state

    split = MonteCarlo(model, scell, nsubs={0: [8]}, seed=409)
    split.metropolis(550, temperature=0.5, target_rate=0.35, max_steps_per_batch=71)
    split.metropolis(550, temperature=0.5, target_rate=0.35, continue_chain=True, max_steps_per_batch=29)
    np.testing.assert_array_equal(split.structure.get_sigmas(), expected_sigmas)
    np.testing.assert_array_equal(
        split._rng_np.bit_generator.state["state"]["state"], expected_rng["state"]["state"]
    )
