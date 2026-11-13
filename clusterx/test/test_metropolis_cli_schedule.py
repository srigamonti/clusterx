"""Maintained CLI schedules, physical continuation, and bounded checkpoint output."""

import importlib
import json
import random
import subprocess
import sys
from copy import deepcopy
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import toml
from ase import Atoms
from ase.io import read, write
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo


cli = importlib.import_module("clusterx.cli.metropolis")


@pytest.fixture
def system(tmp_path):
    atoms = Atoms("HHeLi", positions=[[0, 0, 0], [0.3, 0.2, 0.1], [0.6, 0.5, 0.4]],
                  cell=[1, 3, 4], pbc=[True, False, False])
    # Two substitutional sublattices and one spectator per parent cell.
    plat = ParentLattice(atoms, numbers=[[1, 7], [2, 10], [3]])
    pool = ClustersPool(plat, npoints=[1, 2], radii=[0, 1.01])
    model = Model(CorrelationsCalculator("trigonometric", plat, pool), "energy",
                  ecis=np.linspace(-0.3, 0.2, len(pool)))
    scell = SuperCell(plat, (8, 1, 1))
    plat.serialize(tmp_path / "plat.json")
    return plat, scell, model


def options():
    return dict(task="runmc", plat_filepath="plat.json", model_filepath="model.pickle",
                sc_shape=[8, 1, 1], n_substitutions={0: [4], 1: [4]}, random_seed=109,
                temperature=0.3, energy_factor=1.0, boltzmann_constant=1.0,
                serialize=False, predict_swap=True, reduce_structure=True)


def assert_rng_equal(left, right):
    np.testing.assert_array_equal(left["state"]["state"], right["state"]["state"])
    assert left["has_uint32"] == right["has_uint32"]
    assert left["uinteger"] == right["uinteger"]


@pytest.mark.parametrize("degree", [1, 2])
def test_schedule_matches_uninterrupted_sampling_and_reuses_state(system, monkeypatch, degree):
    _, scell, model = system
    if degree == 2:
        features = np.random.default_rng(7).uniform(-1, 1, size=(100, len(model.ecis)))
        target = features @ model.ecis + 0.7 * features[:, -1] ** 2
        model = Model(model.corrc, "energy", estimator=make_pipeline(
            PolynomialFeatures(2), Ridge(alpha=0.001)
        ).fit(features, target))
    model.serialize("model.pickle")
    baseline = MonteCarlo(model, scell, nsubs={0: [4], 1: [4]}, seed=109,
                          predict_swap=True, reduce_structure=True)
    reference = baseline.metropolis(35, temperature=0.3, nonlinear=True, energy_factor=len(scell),
                                    serialize=True, filename="reference.json",
        boltzmann_constant=1.0,
    )
    expected_sigmas = baseline.structure.get_sigmas().copy()
    expected_np = deepcopy(baseline._rng_np.bit_generator.state)
    expected_std = random.getstate()
    initial = reference._trajectory[0]["sigmas"].copy()

    instances, calls, global_correlations = [], [], []
    original_metropolis = MonteCarlo.metropolis
    original_correlations = Model.get_correlations

    def construct(*args, **kwargs):
        instances.append(MonteCarlo(*args, **kwargs))
        return instances[-1]

    def sampling(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original_metropolis(self, *args, **kwargs)

    def correlations(self, *args, **kwargs):
        global_correlations.append(1)
        return original_correlations(self, *args, **kwargs)

    monkeypatch.setattr(cli, "MonteCarlo", construct)
    monkeypatch.setattr(MonteCarlo, "metropolis", sampling)
    monkeypatch.setattr(Model, "get_correlations", correlations)
    result = cli.metropolis(**{**options(), "energy_factor": len(scell)},
                            runs=[{"temperature": 0.3, "n_mc_steps": 20},
                                              {"temperature": 0.3, "n_mc_steps": 15}],
                            nonlinear=True, max_steps_per_batch=3,
                            save_atoms_every=7, save_atoms_endpoints=True,
                            atoms_filename="schedule.xyz", atoms_mode="x")
    mc = instances[0]
    assert len(instances) == 1
    assert [call["no_of_sampling_steps"] for call in calls] == [20, 15]
    assert [call["continue_chain"] for call in calls] == [False, True]
    assert all(call["energy_factor"] == len(scell) for call in calls)
    assert len(global_correlations) == (1 if degree == 2 else 0)
    np.testing.assert_array_equal(mc.structure.get_sigmas(), expected_sigmas)
    assert_rng_equal(mc._rng_np.bit_generator.state, expected_np)
    assert random.getstate() == expected_std
    assert len(result._trajectory) == 1  # serialize=False retains no accepted-move history.
    assert not list(Path.cwd().glob("mc-trajectory*.json"))

    frames = read("schedule.xyz", index=":")
    assert [frame.info["mc_step"] for frame in frames] == [0, 7, 14, 20, 21, 28, 35]
    assert frames[-1].info["mc_sweeps"] == 35 / 16
    np.testing.assert_array_equal(frames[0].arrays["sigmas"], initial)
    np.testing.assert_array_equal(frames[-1].arrays["sigmas"], expected_sigmas)
    for frame in frames:
        assert np.count_nonzero(frame.numbers == 7) == 4
        assert np.count_nonzero(frame.numbers == 10) == 4
        assert np.all(frame.numbers[scell.get_spectator_sites()] == 3)
        np.testing.assert_array_equal(
            frame.numbers, Structure(scell, sigmas=frame.arrays["sigmas"]).get_decoration()
        )
    if degree == 2:
        cached = mc._chain_state
        full = original_correlations(mc._emodel, mc.structure, reduce_structure=True)
        np.testing.assert_allclose(cached["correlations"], full, atol=1e-12)
        np.testing.assert_allclose(cached["current_energy"], mc._emodel.predict_on_correlations(full), atol=1e-12)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_named_toml_schedule_and_continuation(system, tmp_path, nonlinear):
    _, scell, model = system
    if nonlinear:
        features = np.random.default_rng(71).uniform(-1, 1, size=(100, len(model.ecis)))
        target = features @ model.ecis + 0.5 * features[:, -1] ** 2
        model = Model(model.corrc, "energy", estimator=make_pipeline(
            PolynomialFeatures(2), Ridge(alpha=0.001)
        ).fit(features, target))
    model.serialize("model.pickle")
    configuration = {"do": ["anneal", "continue_anneal"], "unused_fit": {"command": "does_not_exist"}}
    base = options()
    base.pop("temperature")
    base["nonlinear"] = nonlinear
    base["n_substitutions"] = {str(key): value for key, value in base["n_substitutions"].items()}
    for name, output, stages in [
        ("anneal", "anneal.xyz", [{"temperature": 0.7, "n_mc_sweeps": 1},
                                    {"temperature": 0.3, "n_mc_sweeps": 0.5}]),
        ("continue_anneal", "continued.xyz", [{"temperature": 0.2, "n_mc_steps": 5}]),
    ]:
        configuration[name] = {"command": "metropolis", **base, "atoms_filename": output,
                               "save_atoms_endpoints": True, "atoms_mode": "x", "runs": stages,
                               "max_steps_per_batch": 2, "energy_factor": len(scell)}
    configuration["continue_anneal"]["resume_from"] = "anneal.xyz"
    configuration["continue_anneal"]["random_seed"] = 117
    path = tmp_path / "anneal.toml"
    path.write_text(toml.dumps(configuration))
    completed = subprocess.run([sys.executable, "-m", "clusterx.cli.main", "--input_file", str(path)],
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    frames = read("anneal.xyz", index=":")
    resumed = read("continued.xyz", index=":")
    assert [frame.info["temperature"] for frame in frames] == [0.7, 0.7, 0.3]
    assert [frame.info["mc_step"] for frame in frames] == [0, 16, 24]
    np.testing.assert_array_equal(resumed[0].numbers, frames[-1].numbers)
    np.testing.assert_array_equal(resumed[0].positions, frames[-1].positions)
    np.testing.assert_array_equal(resumed[0].cell, frames[-1].cell)
    assert [frame.info["mc_step"] for frame in resumed] == [0, 5]
    assert resumed[0].info["mc_counter_origin"] == "xyz_resume"
    assert not (tmp_path / "mc-trajectory.json").exists()


def test_direct_cli_routes_sampling_and_constructor_options(system, tmp_path):
    _, _, model = system
    model.serialize("model.pickle")
    command = [sys.executable, "-m", "clusterx.cli.main", "metropolis", "-task", "runmc",
               "-scsh", "[8,1,1]", "-nsubs", '{"0":[4],"1":[4]}', "-rs", "109", "-ef", "1", "-k_B", "1",
               "-runs", json.dumps([{"temperature": 0.5, "n_mc_steps": 3},
                                    {"temperature": 0.2, "n_mc_steps": 4}]),
               "-serialize", "false", "-nonlinear", "true", "-reduce", "true", "-batch", "2",
               "-endpoints", "true", "-atoms", "direct.extxyz", "-atoms_mode", "x"]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert [frame.info["mc_step"] for frame in read("direct.extxyz", index=":")] == [0, 3, 7]
    assert not (tmp_path / "mc-trajectory.json").exists()
    # Exercise constructor resume routing through the same direct entry point.
    continued = command[:command.index("-atoms") + 1] + ["direct-continued.xyz", "-atoms_mode", "x",
                                                          "-resume", "direct.extxyz"]
    completed = subprocess.run(continued, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    np.testing.assert_array_equal(read("direct-continued.xyz", index=0).numbers,
                                  read("direct.extxyz", index=-1).numbers)


@pytest.mark.parametrize("extra,match", [
    ({"n_mc_steps": 3, "n_mc_sweeps": 1}, "exactly one"),
    ({"runs": [{"temperature": 1, "n_mc_steps": 1}], "n_mc_steps": 3}, "top level"),
    ({"runs": [{"temperature": 1}]}, "exactly one"),
    ({"n_mc_sweeps": 0.01}, "integer number"),
    ({"runs": [{"temperature": -1, "n_mc_steps": 1}]}, "positive temperature"),
    ({"resume_from": "missing.xyz", "n_mc_steps": 1}, "checkpoint not found"),
    ({"energy_factor": [1]}, "one finite positive number"),
    ({"energy_factor": None}, "energy_factor"),
    ({"boltzmann_constant": None}, "boltzmann_constant"),
    ({"seed": 1}, "unexpected keyword argument"),
    ({"resume_from": "checkpoint.xyz"}, "explicit runs schedule"),
    ({"resume_from": "checkpoint.xyz", "temperature": None, "n_mc_steps": 3}, "explicit runs schedule"),
])
def test_invalid_requests_are_actionable(system, extra, match):
    _, _, model = system
    model.serialize("model.pickle")
    with pytest.raises((ValueError, FileNotFoundError, TypeError), match=match):
        cli.metropolis(**{**options(), **extra})


def test_checkpoint_modes_and_incomplete_resume_frame(system):
    _, scell, model = system
    model.serialize("model.pickle")
    kwargs = {**options(), "n_mc_steps": 3, "save_atoms_endpoints": True, "atoms_filename": "output.xyz"}
    cli.metropolis(**kwargs, atoms_mode="x")
    original = read("output.xyz", index=-1)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.metropolis(**kwargs, atoms_mode="x")
    # An interrupted final frame must not hide earlier complete frames or spoil append.
    buffer = StringIO()
    write(buffer, original, format="extxyz")
    partial = "\n".join(buffer.getvalue().splitlines()[:-1]) + "\nH 0"
    with open("output.xyz", "a") as output:
        output.write(partial)
    cli.metropolis(**kwargs, resume_from="output.xyz", atoms_mode="a")
    frames = read("output.xyz", index=":")
    assert len(frames) == 3  # Original initial/endpoint plus one resumed endpoint.
    np.testing.assert_array_equal(frames[1].numbers, original.numbers)
    assert frames[-1].info["mc_step"] == 3  # Current continuation session only.
    cli.metropolis(**kwargs, atoms_mode="w")
    assert len(read("output.xyz", index=":")) == 2
    invalid = original.copy()
    invalid.cell *= 2
    write("wrong-cell.xyz", invalid)
    with pytest.raises(ValueError, match="cell does not match"):
        cli.metropolis(**kwargs, resume_from="wrong-cell.xyz")
    with open("invalid.xyz", "w") as output:
        output.write("not an XYZ file\n")
    with pytest.raises(ValueError, match="Cannot resume checkpoint"):
        cli.metropolis(**kwargs, resume_from="invalid.xyz")


def test_semigrand_resume_accepts_variable_composition(system):
    _, scell, model = system
    model.serialize("model.pickle")
    write("pristine.xyz", Structure(scell, sigmas=np.zeros(len(scell), dtype=int)).get_atoms())
    result = cli.metropolis(**{**options(), "ensemble": "semigrand-canonical"},
                            chemical_potentials={"0": [0, 0], "1": [0, 0]},
                            resume_from="pristine.xyz", n_mc_steps=3)
    np.testing.assert_array_equal(result._trajectory[0]["sigmas"], 0)


def test_schedule_serialization_keeps_separate_stage_histories(system):
    _, _, model = system
    model.serialize("model.pickle")
    cli.metropolis(**{**options(), "serialize": True}, target_rate=0.35,
                   runs=[{"temperature": 1, "n_mc_steps": 2}, {"temperature": 2, "n_mc_steps": 3}])
    for number, steps in [(1, 2), (2, 3)]:
        with open(f"mc-trajectory_stage{number:04d}.json") as source:
            history = json.load(source)
        assert history["sampling_info"]["number_of_sampling_steps"] == steps
        assert history["sampling_info"]["target_rate"] == 0.35


@pytest.mark.parametrize("nonlinear", [False, True])
def test_named_start_from_can_be_invoked_repeatedly(system, tmp_path, nonlinear):
    _, scell, model = system
    if nonlinear:
        features = np.random.default_rng(75).uniform(-1, 1, (100, len(model.ecis)))
        model.estimator = make_pipeline(PolynomialFeatures(2), Ridge(alpha=0.001)).fit(
            features, features @ model.ecis + 0.4 * features[:, -1]**2,
        )
    model.serialize("model.pickle")
    initial = scell.gen_random_structure(nsubs={0: [4], 1: [4]}, rng=np.random.default_rng(76)).get_atoms()
    earlier = scell.gen_random_structure(nsubs={0: [4], 1: [4]}, rng=np.random.default_rng(77)).get_atoms()
    earlier.info["temperature"], initial.info["temperature"] = 600, 300
    write("300K.xyz", [earlier, initial], format="extxyz")
    source_bytes = Path("300K.xyz").read_bytes()
    configuration = {"do": ["anneal_250K"], "anneal_250K": {
        "command": "metropolis", **options(), "start_from": "300K.xyz", "atoms_filename": "250K.xyz",
        "n_substitutions": {"0": [4], "1": [4]},
        "atoms_mode": "a", "save_atoms_endpoints": True, "nonlinear": nonlinear,
        "runs": [{"temperature": 250, "n_mc_steps": 4}, {"temperature": 200, "n_mc_steps": 3}],
    }}
    path = tmp_path / "reusable.toml"
    path.write_text(toml.dumps(configuration))
    command = [sys.executable, "-m", "clusterx.cli.main", "--input_file", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "does not exist; loading start_from=" in completed.stdout
    first = read("250K.xyz", index=":")
    assert len(first) == 3
    np.testing.assert_array_equal(first[0].numbers, initial.numbers)
    # An interrupted tail is repaired only in the output being resumed.
    with open("250K.xyz", "a") as output:
        output.write(f"{len(scell)}\npartial final frame\nH 0")
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "exists; loading resume_from=" in completed.stdout
    resumed = read("250K.xyz", index=":")
    assert len(resumed) == 5  # Two endpoints per invocation, with no duplicated loaded frame.
    assert [frame.info["mc_step"] for frame in resumed] == [0, 4, 7, 4, 7]
    for previous, actual in zip(first, resumed):
        np.testing.assert_array_equal(actual.numbers, previous.numbers)
    assert Path("300K.xyz").read_bytes() == source_bytes
    for frame in resumed:
        assert np.count_nonzero(frame.numbers == 7) == 4
        assert np.count_nonzero(frame.numbers == 10) == 4
        assert np.all(frame.numbers[scell.get_spectator_sites()] == 3)


def test_direct_start_from_and_distinct_resume_source(system):
    _, scell, model = system
    model.serialize("model.pickle")
    initial = scell.gen_random_structure(nsubs={0: [4], 1: [4]}, rng=np.random.default_rng(78)).get_atoms()
    later = scell.gen_random_structure(nsubs={0: [4], 1: [4]}, rng=np.random.default_rng(79)).get_atoms()
    write("300K.xyz", initial, format="extxyz")
    write("restart.xyz", later, format="extxyz")
    source_bytes = {name: Path(name).read_bytes() for name in ("300K.xyz", "restart.xyz")}
    command = [sys.executable, "-m", "clusterx.cli.main", "metropolis", "-task", "runmc",
               "-scsh", "[8,1,1]", "-nsubs", '{"0":[4],"1":[4]}', "-ef", "1", "-k_B", "1",
               "-temp", "250", "-n_steps", "2", "-serialize", "false", "-reduce", "true",
               "-start", "300K.xyz", "-resume", "restart.xyz", "-atoms", "250K.xyz",
               "-atoms_mode", "a", "-endpoints", "true"]
    for index in range(2):
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        frames = read("250K.xyz", index=":")
        np.testing.assert_array_equal(frames[0 if index == 0 else 2].numbers,
                                      initial.numbers if index == 0 else later.numbers)
        assert len(frames) == 2 * (index + 1)
    assert all(Path(name).read_bytes() == content for name, content in source_bytes.items())


def test_cli_start_from_requires_explicit_length(system):
    _, _, model = system
    model.serialize("model.pickle")
    with pytest.raises(ValueError, match="explicit runs schedule"):
        cli.metropolis(**options(), start_from="300K.xyz")
