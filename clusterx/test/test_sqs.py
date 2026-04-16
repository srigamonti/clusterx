"""Test generation of special quasi-random structures (SQS)."""

import pytest
from ase.build import bulk
import numpy as np

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator


@pytest.fixture
def plat():
    pri = bulk("H", crystalstructure="sc", a=1.0)
    sub = bulk("He", crystalstructure="sc", a=1.0)
    return ParentLattice(atoms=pri, substitutions=[sub], pbc=(1, 1, 1))


@pytest.fixture
def cpool(plat):
    return ClustersPool(plat, npoints=[1, 2, 3], radii=[0, 1.1, 1.1])


@pytest.fixture
def corrc(plat, cpool):
    return CorrelationsCalculator("trigonometric", plat, cpool)


def test_sqs(plat, corrc):
    """Test generation of SQS."""
    n = 10
    p = [n, n, n]
    scell = SuperCell(plat, p)
    for _ in range(10):
        structure = scell.gen_random_structure(
            nsubs=int(np.prod(scell.get_index()) / 2)
        )
        correlations = corrc.get_cluster_correlations(structure)
        print(correlations)
