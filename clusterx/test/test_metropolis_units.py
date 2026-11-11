"""Explicit physical inputs and normalization-invariant Metropolis sampling."""

import importlib
import json
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo, MonteCarloTrajectory


cli = importlib.import_module("clusterx.cli.metropolis")


@pytest.fixture
def system():
    plat = ParentLattice(Atoms("H", positions=[[0, 0, 0]], cell=[1, 3, 4], pbc=[True, False, False]),
                         numbers=[[1, 7]])
    pool = ClustersPool(plat, npoints=[1, 2], radii=[0, 1.01])
    corrc = CorrelationsCalculator("binary-linear", plat, pool)
    return plat, SuperCell(plat, (8, 1, 1)), Model(corrc, "energy", ecis=np.array([0.25, -0.5]))


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("predict_swap,reduce", [(False, False), (True, False), (True, True)])
def test_equivalent_model_normalizations_preserve_sampling(system, nonlinear, predict_swap, reduce):
    _, scell, base = system
    if nonlinear:
        x = np.random.default_rng(17).uniform(0, 1, size=(80, 2))
        y = 0.25 * x[:, 0] - 0.5 * x[:, 1] + 0.75 * x[:, 1]**2
        pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(x, y)
        # Exact rational coefficients remove irrelevant fit noise from zero-energy moves.
        final = pipeline.steps[-1][1]
        final.coef_ = np.round(final.coef_ * 4) / 4
        final.intercept_ = np.round(final.intercept_ * 4) / 4
        base = Model(base.corrc, "energy", estimator=pipeline)

    references = []
    # Each representation yields the same total energy in the chosen energy unit.
    for model_multiplier, energy_factor in [(1, len(scell)), (1000, len(scell) / 1000), (len(scell), 1)]:
        model = deepcopy(base)
        if nonlinear:
            final = model.estimator.steps[-1][1]
            final.coef_ *= model_multiplier
            final.intercept_ *= model_multiplier
        else:
            model.ecis *= model_multiplier
        mc = MonteCarlo(model, scell, nsubs={0: [4]}, seed=83,
                        predict_swap=predict_swap, reduce_structure=reduce)
        trajectory = mc.metropolis(
            60, energy_factor=energy_factor, temperature=0.3, boltzmann_constant=1,
            nonlinear=nonlinear, serialize=True, filename=f"units-{model_multiplier}.json",
            max_steps_per_batch=7,
        )
        references.append((mc.structure.get_sigmas().copy(), mc._rng_np.bit_generator.state,
                           random.getstate(), trajectory.get_sampling_step_nos(),
                           trajectory.get_model_total_energies()))
    for result in references[1:]:
        np.testing.assert_array_equal(result[0], references[0][0])
        np.testing.assert_array_equal(result[1]["state"]["state"], references[0][1]["state"]["state"])
        assert result[2] == references[0][2]
        np.testing.assert_array_equal(result[3], references[0][3])
        np.testing.assert_allclose(result[4], references[0][4], atol=1e-12)


@pytest.mark.parametrize("predict_swap,reduce", [(False, False), (True, False), (True, True)])
def test_semigrand_shortcut_uses_scaled_total_change(system, predict_swap, reduce):
    plat, scell, _ = system
    pool = ClustersPool(plat, npoints=[1], radii=[0])
    model = Model(CorrelationsCalculator("binary-linear", plat, pool), "energy", ecis=np.array([4.0]))
    mc = MonteCarlo(model, scell, ensemble="semigrand-canonical", seed=3,
                    predict_swap=predict_swap, reduce_structure=reduce,
                    chemical_potentials={0: [0, -1]})
    mc._rng_np = SimpleNamespace(random=lambda: 0.99)
    # A flip has raw delta E=.5, scaled delta E=5, and additive cost=-1.
    # The combined +4 must be rejected for this draw (exp(-4) < .99).
    mc.metropolis(1, energy_factor=10, temperature=1, boltzmann_constant=1,
                  initial_decoration=[1] * len(scell))
    np.testing.assert_array_equal(mc.structure.get_sigmas(), 0)


@pytest.mark.parametrize("parameter", ["energy_factor", "temperature", "boltzmann_constant"])
def test_sampling_requires_explicit_physical_inputs(system, parameter):
    _, scell, model = system
    mc = MonteCarlo(model, scell, nsubs={0: [4]})
    arguments = dict(energy_factor=1, temperature=1, boltzmann_constant=1)
    arguments.pop(parameter)
    with pytest.raises(TypeError, match=parameter):
        mc.metropolis(1, **arguments)
    with pytest.raises(ValueError, match=parameter):
        cli.metropolis(task="runmc", **arguments)


@pytest.mark.parametrize("parameter", ["energy_factor", "boltzmann_constant"])
@pytest.mark.parametrize("value", [None, [1], (1,), np.array(1), True, 0, -1, np.nan, np.inf, "1"])
def test_scalar_physical_inputs_are_validated(system, parameter, value):
    _, scell, model = system
    arguments = dict(energy_factor=1, temperature=1, boltzmann_constant=1)
    arguments[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        MonteCarlo(model, scell, nsubs={0: [4]}).metropolis(1, **arguments)
    with pytest.raises(ValueError, match=parameter):
        cli.metropolis(task="runmc", **arguments)


@pytest.mark.parametrize("retired", ["scale_factor", "energy_scale_factor", "energy_normalization"])
def test_removed_options_fail_instead_of_becoming_metadata(system, retired):
    _, scell, model = system
    options = dict(energy_factor=1, temperature=1, boltzmann_constant=1)
    with pytest.raises(TypeError, match=retired):
        MonteCarlo(model, scell).metropolis(1, **options, **{retired: [1]})
    with pytest.raises(TypeError, match=retired):
        cli.metropolis(task="runmc", **options, **{retired: [1]})


def test_total_energy_and_heat_capacity_use_one_factor(system, tmp_path):
    _, scell, _ = system
    reference = None
    for factor in (8, 1):
        raw = np.array([0.0, 8.0, 8.0, 0.0]) / factor
        trajectory = MonteCarloTrajectory(scell, energy_factor=factor, no_of_sampling_steps=3,
                                         temperature=2, boltzmann_constant=0.5, ensemble="canonical")
        for step, energy in enumerate(raw):
            trajectory.add_step(step, energy, [], [], sigmas=np.zeros(len(scell)) if step == 0 else None)
        np.testing.assert_array_equal(trajectory.get_energies(), raw)
        np.testing.assert_array_equal(trajectory.get_model_total_energies(), [0, 8, 8, 0])
        assert trajectory.get_model_total_energy(1) == 8
        assert trajectory.calculate_average_property("U") == 4
        assert trajectory.calculate_average_property("C_p") == 8  # Var=16, k_B*T^2=2.
        path = tmp_path / f"energy-{factor}.json"
        trajectory.serialize(path)
        saved = json.loads(path.read_text())
        assert saved["sampling_info"]["energy_factor"] == factor
        restored = MonteCarloTrajectory(scell, filename=path, read=True)
        np.testing.assert_array_equal(restored.get_model_total_energies(), [0, 8, 8, 0])
        assert restored.calculate_average_property("C_p") == 8
        reference = saved

    # Older trajectories require a user-declared factor, never an inferred conversion.
    reference["sampling_info"].pop("energy_factor")
    path = tmp_path / "older.json"
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="supply it explicitly"):
        MonteCarloTrajectory(scell, filename=path, read=True)
    restored = MonteCarloTrajectory(scell, filename=path, read=True, energy_factor=1)
    np.testing.assert_array_equal(restored.get_model_total_energies(), [0, 8, 8, 0])
