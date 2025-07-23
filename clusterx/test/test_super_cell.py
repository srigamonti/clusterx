"""Tests for the SuperCell class."""

import pytest
from ase import Atoms
import numpy as np

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell


@pytest.fixture
def parent_lattice():
    cell = [[3, 0, 0], [0, 1, 0], [0, 0, 5]]
    positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    pbc = [True, True, False]
    pri = Atoms(["H", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(["C", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(["H", "He", "H"], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(["H", "N", "H"], positions=positions, cell=cell, pbc=pbc)

    return ParentLattice(pri, substitutions=[su1, su2, su3], pbc=pbc)


@pytest.fixture
def super_cell(parent_lattice):
    return SuperCell(parent_lattice, [2, 2, 2])


@pytest.fixture
def parent_lattice_cubic():
    # parent lattice with single site and one substitution
    cell = [1, 1, 1]
    positions = [[0, 0, 0]]
    pbc = [True, True, True]

    pri = Atoms(["Cu"], positions=positions, cell=cell, pbc=pbc)
    sub = Atoms(["Au"], positions=positions, cell=cell, pbc=pbc)
    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)
    return plat


@pytest.fixture
def super_cell_cubic(parent_lattice_cubic):
    return SuperCell(parent_lattice_cubic, [2, 2, 2])


def test_symmetries(parent_lattice_cubic, super_cell_cubic):
    sc_sym = super_cell_cubic.get_symmetry_table()
    assert len(sc_sym) == 3072


def test_translations(super_cell_cubic):
    _ = super_cell_cubic.get_internal_translations()
    _ = super_cell_cubic.get_transformation()


def test_non_diagonal(parent_lattice_cubic):
    scell = SuperCell(parent_lattice_cubic, [[1, 1, 0], [-1, 1, 0], [0, 0, 2]])
    translations = scell.get_internal_translations()
    print(translations)


def test_serialize_load(super_cell):
    super_cell.serialize()
    _ = SuperCell(filepath="scell.json")


def test_super_cell_init(parent_lattice):
    """Test the initialization of the SuperCell class."""
    scell = SuperCell(parent_lattice, [1, 1, 1])
    assert scell.get_parent_lattice() == parent_lattice
    np.testing.assert_array_equal(scell.get_transformation(), np.diag([1, 1, 1]))


def test_methods(super_cell):
    """Test the methods of the SuperCell class."""
    assert super_cell.get_parent_lattice() is not None
    super_cell.compute_sym_perm()
    assert super_cell.get_sym_perm() is not None
    assert super_cell.get_sym() is not None
    assert super_cell.get_sym_platt() is not None
    assert super_cell.get_internal_translations() is not None


def test_gen_random_structure(super_cell):
    """Test the generation of a random structure."""
    random_structure = super_cell.gen_random_structure()
    assert isinstance(random_structure, Atoms)
    assert len(random_structure) == 24
    assert sum(random_structure.get_pbc()) == 2
