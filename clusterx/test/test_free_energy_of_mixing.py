"""Finite canonical entropy, energy origins, and thermodynamic integration."""

import importlib
import itertools
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import toml
from ase import Atoms
from scipy.special import logsumexp
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo, compute_free_energy_of_mixing


cli = importlib.import_module("clusterx.cli.compute_free_energy_of_mixing")


@pytest.fixture
def system():
    atoms = Atoms("HHe", positions=[[0, 0, 0], [0.3, 0.2, 0.1]],
                  cell=[1, 3, 4], pbc=[True, False, False])
    plat = ParentLattice(atoms, numbers=[[1, 7], [2]])
    pool = ClustersPool(plat, npoints=[1], radii=[0])
    assert len(pool) == 1
    model = Model(CorrelationsCalculator("binary-linear", plat, pool), "energy", ecis=np.array([0.75]))
    return plat, SuperCell(plat, (8, 1, 1)), model


def settings(scell, **kwargs):
    return dict(energy_factor=len(scell), boltzmann_constant=0.5, n_mc_steps=24, n_mc_eq=5,
                n_reference_samples=12, n_blocks=3, random_seed=87, max_steps_per_batch=3, **kwargs)


def polynomial_model(model, degree):
    x = np.linspace(0, 1, 40).reshape(-1, 1)
    y = 1.2 + 0.75 * x[:, 0]
    if degree >= 2:
        y -= 0.4 * x[:, 0]**2
    if degree >= 3:
        y += 0.1 * x[:, 0]**3
    pipeline = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(x, y)
    return Model(model.corrc, "energy", estimator=pipeline)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("predict_swap,reduce", [(False, False), (True, False), (True, True)])
def test_configuration_independent_energy_and_pure_limits(system, degree, predict_swap, reduce):
    _, scell, model = system
    model = polynomial_model(model, degree)
    result = compute_free_energy_of_mixing(
        model, scell, [0, 0.25, 0.5, 1], [0.5, 2, 1],
        **settings(scell, predict_swap=predict_swap, reduce_structure=reduce),
    )
    assert result["settings"]["n_sites"] == 8
    assert result["settings"]["n_atoms"] == 16
    assert result["settings"]["species_b"] == 7
    assert result["references"][1]["beta"] == 0
    for row in result["results"]:
        x, temperature = row["composition"], row["temperature"]
        # Spectators make conversion from model eV/atom to eV/binary-site a factor of two.
        energy = 1.2 + 0.75*x - (0.4*x*x if degree >= 2 else 0) + (0.1*x**3 if degree >= 3 else 0)
        e_b = 1.95 - (0.4 if degree >= 2 else 0) + (0.1 if degree >= 3 else 0)
        mixing_energy = 2 * (energy - (1-x)*1.2 - x*e_b)
        entropy = -0.5 * temperature * math.log(math.comb(8, row["n_b"])) / 8
        assert row["mean_energy_per_site"] == pytest.approx(2 * energy, abs=1e-11)
        assert row["free_energy_of_mixing_per_site"] == pytest.approx(mixing_energy + entropy, abs=1e-11)
        assert row["free_energy_of_mixing_per_site_stderr"] < 1e-11
        if x in (0, 1):
            assert row["free_energy_of_mixing_per_site"] == 0
            assert row["n_samples"] == 0
    assert [r["temperature"] for r in result["results"][:3]] == [2, 1, 0.5]


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("predict_swap,reduce", [(False, False), (True, False), (True, True)])
def test_energy_observer_includes_rejections_and_preserves_sampling(system, degree, predict_swap, reduce):
    plat, scell, _ = system
    pool = ClustersPool(plat, npoints=[1, 2], radii=[0, 1.01])
    model = Model(CorrelationsCalculator("binary-linear", plat, pool), "energy", ecis=np.array([0.5, -1.0]))
    if degree == 2:
        features = np.random.default_rng(1).uniform(0, 1, (60, 2))
        target = 0.8 + features[:, 0] - features[:, 1] + 0.5*features[:, 1]**2
        model = Model(model.corrc, "energy", estimator=make_pipeline(
            PolynomialFeatures(2), LinearRegression()).fit(features, target))
    original = deepcopy(model)
    full = deepcopy(model)
    full.reset_mc(False)
    mc = MonteCarlo(model, scell, nsubs={0: [4]}, seed=11, predict_swap=predict_swap, reduce_structure=reduce)
    observations = []

    def observe(step, energy):
        assert energy == pytest.approx(3 * full.predict(mc.structure), abs=1e-10)
        assert np.count_nonzero(mc.structure.get_decoration() == 7) == 4
        assert np.all(mc.structure.get_decoration()[scell.get_spectator_sites()] == 2)
        observations.append((step, energy))

    options = dict(energy_factor=3, temperature=0.05, boltzmann_constant=1, nonlinear=True)
    # Attach the observer after equilibration, then reuse its energy origin after an unobserved call.
    mc.metropolis(11, **options)
    observed = mc.metropolis(60, **options, continue_chain=True, energy_observer=observe,
                             serialize=True, filename="observed.json", max_steps_per_batch=7)
    mc.metropolis(4, **options, continue_chain=True)

    def observe_rescaled(_step, energy):
        np.testing.assert_allclose(energy, 2 * full.predict(mc.structure), atol=1e-10)

    mc.metropolis(9, **{**options, "energy_factor": 2}, continue_chain=True,
                  energy_observer=observe_rescaled)
    assert len(observations) == 60
    assert len(observed._trajectory) < 60  # Rejections must also contribute to the average.
    observed_std_state = mc._rng_std.getstate()
    baseline = MonteCarlo(original, scell, nsubs={0: [4]}, seed=11,
                          predict_swap=predict_swap, reduce_structure=reduce)
    baseline.metropolis(11, **options)
    expected = baseline.metropolis(60, **options, continue_chain=True, serialize=True, filename="baseline.json")
    baseline.metropolis(4, **options, continue_chain=True)
    baseline.metropolis(9, **{**options, "energy_factor": 2}, continue_chain=True)
    np.testing.assert_array_equal(mc.structure.get_sigmas(), baseline.structure.get_sigmas())
    np.testing.assert_array_equal(observed.get_sampling_step_nos(), expected.get_sampling_step_nos())
    observed_state = mc._rng_np.bit_generator.state
    baseline_state = baseline._rng_np.bit_generator.state
    np.testing.assert_array_equal(observed_state["state"]["state"], baseline_state["state"]["state"])
    assert observed_state["uinteger"] == baseline_state["uinteger"]
    assert observed_state["has_uint32"] == baseline_state["has_uint32"]
    assert observed_std_state == baseline._rng_std.getstate()


def test_interacting_ring_against_exact_partition_sum(system):
    plat, _, _ = system
    scell = SuperCell(plat, (4, 1, 1))
    pool = ClustersPool(plat, npoints=[2], radii=[1.01])
    model = Model(CorrelationsCalculator("binary-linear", plat, pool), "energy", ecis=np.array([1.0]))
    energies = []
    for pair in itertools.combinations(scell.get_substitutional_sites(), 2):
        sigmas = np.zeros(len(scell), dtype=int)
        sigmas[list(pair)] = 1
        energies.append(len(scell) * model.predict(Structure(scell, sigmas=sigmas)))
    energies = np.array(energies)
    # At fixed NB=2 there are four adjacent and two separated configurations.
    np.testing.assert_allclose(np.sort(energies), [0, 0, 2, 2, 2, 2])
    betas = np.linspace(0.05, 1, 12)
    result = compute_free_energy_of_mixing(
        model, scell, 0.5, 1/betas, energy_factor=len(scell), boltzmann_constant=1,
        n_mc_steps=3000, n_mc_eq=200, n_reference_samples=2000, n_blocks=10, random_seed=19,
        predict_swap=True, reduce_structure=True,
    )
    for row in result["results"]:
        beta = row["beta"]
        expected_mean = np.dot(np.exp(-beta*energies), energies) / np.exp(-beta*energies).sum() / 4
        expected_free = -logsumexp(-beta*energies) / (beta*4) - 1  # Pure-B energy/site=2.
        assert row["mean_energy_per_site"] == pytest.approx(expected_mean, abs=0.035)
        assert row["free_energy_of_mixing_per_site"] == pytest.approx(expected_free, abs=0.025)
        assert row["free_energy_of_mixing_per_site_stderr"] > 0
    # The finite canonical beta=0 pair correlation is 1/6, not x**2=1/4.
    assert result["references"][0]["mean_energy_per_site"] == pytest.approx(1/3, abs=0.025)


@pytest.mark.parametrize("values,name", [([0, 1], "temperatures"), ([-1], "temperatures"),
                                        ([1, 1], "temperatures"), ([np.nan], "temperatures"),
                                        ([np.inf], "temperatures"), ([[1]], "temperatures"),
                                        ([True], "temperatures"), ([], "compositions"),
                                        ([0.1], "compositions"), ([0.5, 0.5], "compositions"),
                                        ([-0.1], "compositions"), ([1.1], "compositions")])
def test_grid_validation(system, values, name):
    _, scell, model = system
    grid = dict(compositions=0.5, temperatures=[1, 2])
    grid[name] = values
    with pytest.raises(ValueError, match="temperatures|compositions|composition"):
        compute_free_energy_of_mixing(model, scell, **grid, **settings(scell))


@pytest.mark.parametrize("name,value", [("n_mc_steps", 4), ("n_mc_eq", -1), ("sample_interval", 0),
                                      ("n_reference_samples", 1), ("n_blocks", 1),
                                      ("n_mc_steps", 24.5), ("n_blocks", True), ("max_steps_per_batch", 0)])
def test_sampling_validation(system, name, value):
    _, scell, model = system
    options = settings(scell)
    options[name] = value
    with pytest.raises(ValueError, match=name + "|two samples"):
        compute_free_energy_of_mixing(model, scell, 0.5, 1, **options)


def test_temperature_order_output_and_bounded_storage(system, tmp_path, monkeypatch):
    _, scell, model = system
    calls = []
    original = MonteCarlo.metropolis

    def sampling(self, *args, **kwargs):
        trajectory = original(self, *args, **kwargs)
        assert not kwargs["serialize"]
        assert len(trajectory._trajectory) == 1
        calls.append(kwargs)
        return trajectory

    monkeypatch.setattr(MonteCarlo, "metropolis", sampling)
    path = tmp_path / "free.json"
    result = compute_free_energy_of_mixing(model, scell, 0.5, [1, 2], filename=path, **settings(scell))
    inverse = compute_free_energy_of_mixing(deepcopy(model), scell, 0.5, [2, 1], **settings(scell))
    assert result["results"] == inverse["results"]
    assert json.loads(path.read_text()) == result
    assert not calls[0]["continue_chain"]
    assert all(call["continue_chain"] for call in calls[1:4])
    assert all(call["max_steps_per_batch"] == 3 for call in calls)
    with pytest.raises(FileExistsError, match="overwrite"):
        compute_free_energy_of_mixing(model, scell, 0.5, 1, filename=path, **settings(scell))
    compute_free_energy_of_mixing(model, scell, 0.5, 1, filename=path, overwrite=True, **settings(scell))


@pytest.mark.parametrize("degree", [1, 2])
def test_observer_cadence_and_cached_energy_origin(system, monkeypatch, degree):
    _, scell, model = system
    model = polynomial_model(model, degree)
    predictions, observations = [], []
    original = model.predict

    def predict(*args, **kwargs):
        predictions.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "predict", predict)
    mc = MonteCarlo(model, scell, nsubs={0: [4]}, seed=4, predict_swap=True, reduce_structure=True)
    options = dict(energy_factor=len(scell), temperature=1, boltzmann_constant=1, nonlinear=True)
    mc.metropolis(7, **options)
    assert not predictions  # Affine reduced sampling needs no absolute energy origin yet.
    for _ in range(2):
        mc.metropolis(7, **options, continue_chain=True, observation_interval=3,
                      energy_observer=lambda step, energy: observations.append((step, energy)))
    assert len(predictions) == (1 if degree == 1 else 0)
    assert [step for step, _ in observations] == [3, 6, 3, 6]


def test_unsupported_chemistry_and_nonlinear_settings(system):
    _, scell, model = system
    ternary = ParentLattice(Atoms("H", cell=[1, 3, 4], pbc=[True, False, False]), numbers=[[1, 2, 7]])
    pool = ClustersPool(ternary, npoints=[1], radii=[0])
    ternary_model = Model(CorrelationsCalculator("trigonometric", ternary, pool), "energy",
                          ecis=np.zeros(len(pool)))
    with pytest.raises(ValueError, match="binary substitutional"):
        compute_free_energy_of_mixing(ternary_model, SuperCell(ternary, (8, 1, 1)), 0.5, 1, **settings(scell))
    with pytest.raises(ValueError, match="parent lattices"):
        compute_free_energy_of_mixing(model, SuperCell(ternary, (8, 1, 1)), 0.5, 1, **settings(scell))
    with pytest.raises(ValueError, match="nonlinear=True"):
        compute_free_energy_of_mixing(polynomial_model(model, 2), scell, 0.5, 1,
                                     **settings(scell, nonlinear=False))


def test_model_cache_file_restored_after_prediction_failure(system, monkeypatch):
    _, scell, model = system
    model.corrc.pickle_file = "must-not-be-written.pickle"

    def invalid(*_args, **_kwargs):
        return np.nan

    monkeypatch.setattr(model, "predict", invalid)
    with pytest.raises(ValueError, match="finite energy"):
        compute_free_energy_of_mixing(model, scell, 0.5, 1, **settings(scell))
    assert model.corrc.pickle_file == "must-not-be-written.pickle"
    assert not Path(model.corrc.pickle_file).exists()


def test_direct_cli_and_named_toml(system, tmp_path):
    plat, scell, model = system
    plat.serialize("plat.json")
    model.serialize("model.pickle")
    artifacts = {path: path.read_bytes() for path in Path.cwd().glob("*.pickle")}
    command = [sys.executable, "-m", "clusterx.cli.main"]
    direct = subprocess.run(command + [
        "compute_free_energy_of_mixing", "-mofp", "model.pickle", "-plfp", "plat.json",
        "-scsh", "[8,1,1]", "-x", "[0,0.5,1]", "-temps", "[2,1]", "-ef", str(len(scell)),
        "-k_B", "0.5", "-n_steps", "24", "-n_eq", "5", "-n_ref", "12", "-blocks", "3",
        "-rs", "87", "-reduce", "true", "-out", "direct.json",
    ], capture_output=True, text=True)
    assert direct.returncode == 0, direct.stdout + direct.stderr
    configuration = {"do": ["mixing"], "mixing": dict(
        command="compute_free_energy_of_mixing", sc_shape=[8, 1, 1], compositions=[0.0, 0.5, 1.0],
        temperature_start=1, temperature_stop=2, n_temperatures=2,
        reduce_structure=True, output_filepath="named.json", **settings(scell))}
    path = tmp_path / "input.toml"
    path.write_text(toml.dumps(configuration))
    named = subprocess.run(command + ["--input_file", str(path)], capture_output=True, text=True)
    assert named.returncode == 0, named.stdout + named.stderr
    direct_results = json.loads(Path("direct.json").read_text())["results"]
    assert direct_results == json.loads(Path("named.json").read_text())["results"]
    assert not list(tmp_path.glob("*trajectory*"))
    assert all(path.read_bytes() == content for path, content in artifacts.items())


@pytest.mark.parametrize("kwargs", [{}, {"temperatures": [1], "temperature_start": 1},
                                   {"temperature_start": 1, "n_temperatures": 2},
                                   {"temperature_start": 1, "temperature_stop": 1, "n_temperatures": 2}])
def test_cli_temperature_range_validation(kwargs):
    with pytest.raises(ValueError, match="temperature|range"):
        cli.compute_free_energy_of_mixing(energy_factor=1, boltzmann_constant=1, **kwargs)
