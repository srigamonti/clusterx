# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import os
from typing import Optional, Union, List

import pytest
from ase.spacegroup import crystal
from ase.data import atomic_numbers as cn
from ase import Atoms
import numpy as np

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.clusters.cluster import Cluster
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.thermodynamics.monte_carlo import MonteCarlo, MonteCarloTrajectory
from clusterx.thermodynamics.monte_carlo_lite import MonteCarloLite
from clusterx.cli.metropolis import metropolis


@pytest.fixture
def plat():
    # parent lattice with single site and one substitution
    cell = [1, 1, 1]
    positions = [[0, 0, 0]]
    pbc = [True, True, False]

    pri = Atoms(["Cu"], positions=positions, cell=cell, pbc=pbc)
    sub = Atoms(["Au"], positions=positions, cell=cell, pbc=pbc)
    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)
    return plat


@pytest.fixture
def plat_bulk():
    # parent lattice with single site and one substitution
    cell = [1, 1, 1]
    positions = [[0, 0, 0]]
    pbc = [True, True, True]

    pri = Atoms(["Cu"], positions=positions, cell=cell, pbc=pbc)
    sub = Atoms(["Au"], positions=positions, cell=cell, pbc=pbc)
    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)
    return plat


@pytest.fixture
def plat_perov():
    """Perovskite mult-sublattice parent lattice"""
    cell = [6, 6, 6]
    positions = [
        [0, 0, 0],
        [3, 3, 3],
        [0, 3, 3],
        [3, 0, 3],
        [3, 3, 0],
    ]
    pbc = [True, True, True]
    pri = Atoms(["Cs", "Pb", "I", "I", "I"], positions=positions, cell=cell, pbc=pbc)
    sub_Sn = Atoms(["Cs", "Sn", "I", "I", "I"], positions=positions, cell=cell, pbc=pbc)
    sub_Br = Atoms(
        ["Cs", "Pb", "Br", "Br", "Br"], positions=positions, cell=cell, pbc=pbc
    )
    plat = ParentLattice(pri, substitutions=[sub_Sn, sub_Br], pbc=pbc)
    return plat


@pytest.fixture
def cpool(plat):
    # one-point clusters and nearest neighbor two-point clusters
    return ClustersPool(plat, npoints=[1, 2], radii=[0, 1.1])


@pytest.fixture
def corrc(plat, cpool):
    return CorrelationsCalculator("trigonometric", plat, cpool)


@pytest.fixture
def model(plat, cpool, corrc):
    # Ising-like model
    ecisE = [0.0, -1.0]
    multT = cpool.get_multiplicities()
    cemodel = Model(corrc, "energy", ecis=np.multiply(ecisE, multT))
    return cemodel


@pytest.fixture
def model_bulk(plat_bulk):
    cpool = ClustersPool(plat_bulk, npoints=[1, 2], radii=[0, 1.1])
    corrc = CorrelationsCalculator("trigonometric", plat_bulk, cpool)
    # Ising-like model
    ecisE = [0.0, -1.0]
    multT = cpool.get_multiplicities()
    cemodel = Model(corrc, "energy", ecis=np.multiply(ecisE, multT))
    return cemodel


@pytest.fixture
def model_multi(plat, cpool, corrc):
    # Model for two properties in parallel
    ecis = np.array([[-1.0, -2.0], [-2.0, -4.0]])
    cemodel = Model(corrc, ["energy", "energy_double"], ecis=ecis)
    return cemodel


@pytest.fixture
def wyckoff_sites():
    x, y, z = 0.185, 0.304, 0.116
    return [
        (0, y, z),  # 24k
        (x, x, x),  # 16i
        (1 / 4.0, 0, 1 / 2.0),  # 6c
        (1 / 4.0, 1 / 2.0, 0),  # 6d
        (0, 0, 0),  # 2a
    ]


@pytest.fixture
def cell_a():
    return 10.5148


@pytest.fixture
def pristine_crystal(wyckoff_sites, cell_a):
    return crystal(
        ["Si", "Si", "Si", "Ba", "Ba"],
        wyckoff_sites,
        spacegroup=223,
        cellpar=[cell_a, cell_a, cell_a, 90, 90, 90],
    )


@pytest.fixture
def sub_Al(wyckoff_sites, cell_a):
    return crystal(
        ["Al", "Al", "Al", "Ba", "Ba"],
        wyckoff_sites,
        spacegroup=223,
        cellpar=[cell_a, cell_a, cell_a, 90, 90, 90],
    )


@pytest.fixture
def sub_X(wyckoff_sites, cell_a):
    return crystal(
        ["X", "X", "X", "Ba", "Ba"],
        wyckoff_sites,
        spacegroup=223,
        cellpar=[cell_a, cell_a, cell_a, 90, 90, 90],
    )


@pytest.fixture
def sub_Sr(wyckoff_sites, cell_a):
    return crystal(
        ["Al", "Al", "Al", "Sr", "Sr"],
        wyckoff_sites,
        spacegroup=223,
        cellpar=[cell_a, cell_a, cell_a, 90, 90, 90],
    )


@pytest.fixture
def plat_binary(pristine_crystal, sub_Al):
    return ParentLattice(atoms=pristine_crystal, substitutions=[sub_Al], pbc=(1, 1, 1))


@pytest.fixture
def plat_full_sub(pristine_crystal, sub_Al, sub_X, sub_Sr):
    return ParentLattice(
        atoms=pristine_crystal, substitutions=[sub_Al, sub_X, sub_Sr], pbc=(1, 1, 1)
    )


@pytest.fixture
def cemodel_binary(pristine_crystal, sub_Al):
    plat = ParentLattice(atoms=pristine_crystal, substitutions=[sub_Al], pbc=(1, 1, 1))

    cpool = ClustersPool(plat)
    cpsc = cpool.get_cpool_scell()
    s = cn["Al"]
    cpool.add_cluster(Cluster([24], [s], cpsc))
    cpool.add_cluster(Cluster([40], [s], cpsc))
    cpool.add_cluster(Cluster([6, 4], [s, s], cpsc))
    cpool.add_cluster(Cluster([37, 32], [s, s], cpsc))
    cpool.add_cluster(Cluster([39, 12], [s, s], cpsc))
    cpool.add_cluster(Cluster([16, 43], [s, s], cpsc))
    cpool.add_cluster(Cluster([35, 11], [s, s], cpsc))
    cpool.add_cluster(Cluster([39, 30], [s, s], cpsc))
    cpool.add_cluster(Cluster([35, 42], [s, s], cpsc))
    cpool.add_cluster(Cluster([18, 43], [s, s], cpsc))

    ecis = [
        -78407.3247588,
        47.164484875,
        47.1673476881,
        47.1569012692,
        0.00851281608144,
        0.0139835351147,
        0.0108175321899,
        0.0101521144776,
        0.00121744613474,
        0.000413664306204,
    ]

    multT = [1, 24, 16, 6, 12, 8, 48, 24, 24, 24]
    corc = CorrelationsCalculator("binary-linear", plat, cpool)
    cemodel = Model(corc, "energy", ecis=np.multiply(ecis, multT))
    return cemodel


@pytest.fixture
def cemodel_full_sub(plat_full_sub):
    cpool = ClustersPool(plat_full_sub, npoints=[1], radii=[0])
    corrc = CorrelationsCalculator("trigonometric", plat_full_sub, cpool)
    mult = cpool.get_multiplicities()
    ecis = [-78407.325, 23.16, 23.15, 23.14, 23.13, 23.12, 23.11, 23.10]
    return Model(corrc, "energy2", ecis=np.multiply(ecis, mult))


@pytest.fixture
def cli_kwargs():
    """Collection of reproducible default arguments."""
    args = {}
    args["task"]: str = "usage"
    args["plat_filepath"]: str = "plat.json"
    args["model_filepath"]: str = "model.pickle"
    args["mcsetup_filepath"]: str = "mc-setup.pickle"
    args["mcrun_filepath"]: str = "mc-run.pickle"
    args["mcrun_filepaths"]: list[str] = None
    args["plotdata_filepath"]: str = None
    args["show_plot"] = True
    args["save_plot"] = False
    args["keep_sigmas"]: Optional[int] = 1
    args["mcscell_filepath"]: str = "mc-scell.pickle"
    args["traj_filepath"]: str = "mc-trajectory.json"
    args["sc_shape"]: Optional[Union[int, List[int], List[List[int]]]] = 2
    args["n_substitutions"]: Optional[Union[int, dict]] = 1
    args["ensemble"]: str = "canonical"
    args["sublattice_indices"]: List[int] | None = None
    args["chemical_potential"]: float = 0.0
    args["chemical_potentials"]: Optional[dict] = None
    args["models_aux_filepaths"]: List[str] = []
    args["no_of_swaps"]: int = 1
    args["predict_swap"]: bool = True
    # metropolis sampling arguments
    args["n_mc_steps"]: int = 100
    args["n_mc_eq"]: int = 1
    args["n_clics"]: int = 1
    args["runs"]: Optional[dict] = None
    args["energy_scale_factor"]: float | None = None
    args["temperature"]: float = 1.0
    args["temperatures"]: Optional[List[float]] = None
    args["boltzmann_constant"]: float = 1.0
    args["initial_decoration"]: Optional[List[int]] = None
    args["target_rate"]: Optional[float] = None
    args["random_seed"]: Optional[int] = None
    return args


@pytest.mark.parametrize(
    "task,filelist",
    [
        ("usage", []),
        ("runmc", []),
        ("setup", ["mc-setup.pickle"]),
        ("run", ["mc-run.pickle"]),
    ],
)
def test_cli(plat, model, cli_kwargs, task, filelist):
    """Perform the specified task from cli function and check that correct
    files are generated."""
    model_filepath = "model.pickle"
    model.serialize(model_filepath)
    plat_filepath = "plat.json"
    plat.serialize(plat_filepath)
    cli_kwargs["task"] = task
    metropolis(**cli_kwargs)
    for filename in filelist:
        print(filename)
        assert os.path.exists(filename)


@pytest.mark.parametrize("model_", ["model", "model_multi"])
def test_metropolis(plat, model_, request):
    # prime example of using multiple fixtures with parametrize
    scell = SuperCell(plat, 4)
    model = request.getfixturevalue(model_)
    mc = MonteCarlo(model, scell, nsubs={0: [4]})
    mc.metropolis(no_of_sampling_steps=10, temperature=600, serialize=True)
    traj = MonteCarloTrajectory(read=True)
    traj.get_lowest_energy_structure()


@pytest.mark.parametrize("reduce", [False, True])
def test_metropolis_reduce(plat_bulk, model_bulk, reduce):
    scell = SuperCell(plat_bulk, 10)
    n_atoms = len(scell)
    mc = MonteCarlo(
        model_bulk,
        scell,
        nsubs={0: [n_atoms // 2]},
        predict_swap=True,
        reduce_structure=reduce,
    )
    mc.metropolis(no_of_sampling_steps=1000, temperature=600, serialize=True)
    traj = MonteCarloTrajectory(read=True)
    traj.get_lowest_energy_structure()


def test_metropolis_perov(plat_perov):
    cpool = ClustersPool(plat_perov, npoints=[1, 2], radii=[0, 6.1])
    cpool.serialize("cpool_perov.json")
    corrc = CorrelationsCalculator("trigonometric", plat_perov, cpool)
    np.random.seed(42)
    ecis = np.random.normal(size=len(cpool))
    cemodel = Model(corrc, "energy", ecis=ecis)

    scell = SuperCell(plat_perov, 10)
    mc = MonteCarlo(
        cemodel,
        scell,
        nsubs=None,
        predict_swap=True,
        reduce_structure=True,
    )
    mc.metropolis(no_of_sampling_steps=1000, temperature=1e-3, serialize=False)


@pytest.mark.parametrize("model_", ["model", "model_multi"])
def test_mclite(plat, model_, request, tmp_path):
    scell = SuperCell(plat, 4)
    n_atoms = len(scell)
    model = request.getfixturevalue(model_)
    mc = MonteCarloLite(model, scell, boltzmann_constant=3e-6)
    mcrun = mc.metropolis(
        n_mc_steps=10,
        temperature=600,
        mcrun_filepath=tmp_path / "mc-run_test.pickle",
        n_substitutions=n_atoms // 2,
        serialize=True,
        random_seed=42,
    )
    assert mcrun.energies  # check existence


@pytest.mark.parametrize("reduce", [False, True])
def test_mclite_reduce(plat_bulk, model_bulk, tmp_path, reduce):
    scell = SuperCell(plat_bulk, 10)
    n_atoms = len(scell)
    mc = MonteCarloLite(
        model_bulk, scell, boltzmann_constant=1, reduce_structure=reduce
    )
    mcrun = mc.metropolis(
        n_mc_steps=10,
        temperature=10,
        mcrun_filepath=tmp_path / "mc-run_test.pickle",
        n_substitutions=n_atoms // 2,
        serialize=True,
        random_seed=42,
    )
    assert mcrun.energies  # check existence


def test_mclite_reduce_large(plat_bulk, model_bulk, tmp_path):
    scell = SuperCell(plat_bulk, 100)
    n_atoms = len(scell)
    mc = MonteCarloLite(model_bulk, scell, boltzmann_constant=1, reduce_structure=True)
    mcrun = mc.metropolis(
        n_mc_steps=1_000,
        temperature=10,
        mcrun_filepath=tmp_path / "mc-run_test.pickle",
        n_substitutions=n_atoms // 2,
        serialize=True,
        random_seed=42,
    )
    assert mcrun.energies  # check existence


def test_reduce_nondiag_raise(plat_bulk, model_bulk):
    p = [[1, 1, 0], [-1, 1, 0], [0, 0, 2]]
    scell = SuperCell(plat_bulk, p)
    with pytest.raises(ValueError):
        MonteCarloLite(model_bulk, scell, reduce_structure=True)


def test_metropolis_clathrate_full_subs(
    plat_full_sub,
    cemodel_full_sub,
):
    scell = SuperCell(plat_full_sub, 2)
    mc = MonteCarlo(
        cemodel_full_sub,
        scell,
        nsubs={0: [112, 16], 1: [0]},
    )
    nmc = 30
    kb = float(3.16681009610757e-6)
    traj = mc.metropolis(
        no_of_sampling_steps=nmc,
        temperature=600,
        boltzmann_constant=kb,
        serialize=True,
        filename="trajectory-ternary.json",
    )
    steps = traj.get_sampling_step_nos()
    traj.get_energies()
    traj.get_sampling_step_entry_at_step(steps[-1])
    traj.get_structure_at_step(steps[-1])
    traj.serialize()
    assert os.path.isfile("trajectory-ternary.json")
