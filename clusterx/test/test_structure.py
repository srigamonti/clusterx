"""Test the Structure class from clusterx.structure"""

import pytest
from ase import Atoms

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure


@pytest.fixture
def parent_lattice():
    cell = [
        [3,0,0],
        [0,1,0],
        [0,0,5]]
    positions = [
        [0,0,0],
        [1,0,0],
        [2,0,0]]
    pbc = [True,True,False]
    pri = Atoms(['H','H','H'], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(['C','H','H'], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(['H','He','H'], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(['H','N','H'], positions=positions, cell=cell, pbc=pbc)

    return ParentLattice(pri, substitutions=[su1,su2,su3], pbc=pbc)


@pytest.fixture
def super_cell(parent_lattice):
    return SuperCell(parent_lattice, [(1,0,0),(0,3,0),(0,0,1)])


@pytest.fixture
def structure(super_cell):
    return Structure(super_cell, decoration=[1]*len(super_cell))


def test_init_scell(structure):
    assert structure.scell is not None