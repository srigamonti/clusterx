"""Test generation of special quasi-random structures (SQS)."""

import pytest
from ase.build import bulk
from ase.visualize import view

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.cli.generate_sqs import generate_sqs, run_annealing


@pytest.fixture
def plat():
    pri = bulk("H", crystalstructure="sc", a=1.0)
    sub = bulk("He", crystalstructure="sc", a=1.0)
    return ParentLattice(atoms=pri, substitutions=[sub], pbc=(1, 1, 1))


@pytest.fixture
def cpool(plat):
    return ClustersPool(plat, npoints=[1, 2, 3], radii=[0, 2.1, 1.5])


@pytest.fixture
def corrc(plat, cpool):
    cpool.print_info()
    return CorrelationsCalculator("trigonometric", plat, cpool)


def test_sqs_correlation(plat, corrc):
    """Test generation of SQS."""
    n = 4
    p = [n, n, n]
    scell = SuperCell(plat, p)
    n_atoms = scell.get_index()
    structure = run_annealing(plat, corrc, p, n_atoms // 2)
    # view(structure.get_atoms())


def test_cli():
    generate_sqs()
