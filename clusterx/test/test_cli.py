"""Test the CLI commands of the clusterx package.

Also demonstrate the use of pytests tmp_path and how to test CLI in general.
"""

import os
import subprocess

import pytest
from ase import Atoms

from clusterx.cli.build_cpool import build_cpool
from clusterx.cli.compute_comat import compute_comat
from clusterx.cli.compute_concentrations import compute_concentrations
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.derivative_structures import DerivativeStructures
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell

@pytest.fixture
def parent_lattice():
    a = 1.0
    positions = [(0, 0, 0), (a / 2, a / 2, a / 2)]
    cell = [(a, 0, 0), (0, a, 0), (0, 0, a)]
    pbc = (True, True, True)
    pri = Atoms("Cu2", positions=positions, cell=cell, pbc=pbc)
    sub = Atoms("Al2", positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)
    return plat


@pytest.fixture
def cpool(parent_lattice):
    # one-point clusters and nearest neighbor two-point clusters
    return ClustersPool(parent_lattice, npoints=[1, 2], radii=[0, 0.9])


@pytest.fixture
def corrc(parent_lattice, cpool):
    return CorrelationsCalculator("trigonometric", parent_lattice, cpool)


@pytest.fixture
def super_cell(parent_lattice):
    return SuperCell(parent_lattice, p=1)


@pytest.fixture
def structure(super_cell):
    return Structure(super_cell, sigmas=[0]*len(super_cell))


@pytest.fixture
def structures_set(parent_lattice, structure):
    sset = StructuresSet(parent_lattice)
    sset.add_structure(structure)
    return sset


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result


def err_str(command, response):
    return f"Command '{command}' failed with exit code {response.returncode}\nStdout: {response.stdout}\nStderr: {response.stderr}"


@pytest.mark.parametrize("command", ["cell -h", "cell --help"])
def test_help(command):
    response = run_command("cell -h")
    assert response.returncode == 0, err_str(command, response)


def test_legacy_summary_alias():
    from clusterx.cli.summarize_derivative_structures import summarize_derivative_structures
    from clusterx.cli.summarize_dsgenerator import summarize_dsgenerator

    assert summarize_dsgenerator is summarize_derivative_structures


@pytest.mark.parametrize("command", ["summarize_derivative_structures", "summarize_dsgenerator"])
def test_summarize_derivative_structures(command, structures_set, tmp_path):
    filepath = tmp_path / "derivative_structures.pickle"
    DerivativeStructures.from_structures_set(structures_set).serialize(filepath=filepath)

    response = subprocess.run(["cell", command, str(filepath)], capture_output=True, text=True)

    assert response.returncode == 0, err_str(command, response)
    assert "DerivativeStructures summary" in response.stdout
    assert "Number of configurations: 1" in response.stdout


def run_build_parent_lattice(path):
    command = (
        f"cell build_parent_lattice --species=Si,Ge --filepath={path / 'plat.json'} "
    )
    response = run_command(command)
    assert response.returncode == 0, err_str(command, response)


def run_build_cpool(path):
    command = (
        "cell build_cpool 2,3 1.0,2.0,3.0 "
        f"-plf {path / 'plat.json'} "
        "--psc=2 "
        "--method=1 "
        f"-cpool_filepath={path / 'cpool.json'} "
        "--vacancy_atomic_number 0 "
    )
    response = run_command(command)
    assert response.returncode == 0, err_str(command, response)


def test_build(tmp_path):
    # this is a workaround, ideally the tests should be independent, but this
    # would only be possible with extra pytest-dependency package
    run_build_parent_lattice(tmp_path)

    # Check if the files were created
    assert os.path.exists(tmp_path / "plat.json")


def test_build_cpool_plat(tmp_path, parent_lattice):
    plat_path = tmp_path / "plat.json"
    cpool_path = tmp_path / "cpool.json"
    parent_lattice.serialize(plat_path)
    build_cpool(
        npoints=[1, 2],  # List[int] or str
        radii=[0, 1.1],  # List[float] or str
        post_manual_select=None,  # Optional[List[int]]
        sset_filepath=None,  # Optional[str]
        plat_filepath=plat_path,  # Optional[str]
        psc=None,  # Supercell definition
        method=0,  # int
        cpool_filepath=str(cpool_path),  # str
        vacancy_atomic_number=0,  # int
    )
    assert os.path.exists(str(cpool_path))


def test_build_cpool_sset(tmp_path, structures_set):
    sset_path = tmp_path / "sset.xyz"
    structures_set.serialize(sset_path)
    cpool_path = tmp_path / "cpool.json"
    build_cpool(
        npoints=[1, 2],  # List[int] or str
        radii=[0, 1.1],  # List[float] or str
        post_manual_select=None,  # Optional[List[int]]
        sset_filepath=sset_path,  # Optional[str]
        plat_filepath=None,  # Optional[str]
        psc=None,  # Supercell definition
        method=0,  # int
        cpool_filepath=str(cpool_path),  # str
        vacancy_atomic_number=0,  # int
    )
    assert os.path.exists(str(cpool_path))


def test_compute_comat(tmp_path, corrc, structures_set):
    corrc_path = tmp_path / "corrc.pickle"
    corrc.serialize(corrc_path)
    sset_path = str(tmp_path / "sset.json")
    structures_set.serialize(sset_path)
    comat_path = tmp_path / "comat.npz"
    compute_comat(
        ccalc_filepath=corrc_path,
        sset_filepath=sset_path,
        sset_filepaths=None,
        dss_filepath=None,
        dss_filepaths=None,
        comat_filepath=comat_path,
        property_name=None,
        update_ccalc=False,
    )
    assert os.path.exists(comat_path)


def test_compute_concentrations(tmp_path, structures_set):
    sset_path = str(tmp_path / "sset.json")
    structures_set.serialize(sset_path)
    conc_path = tmp_path / "conc.npz"
    compute_concentrations(
        sset_filepath=sset_path,
        site_type=0,
        sigma=1,
        conc_filepath=conc_path,
    )
    assert os.path.exists(conc_path)
