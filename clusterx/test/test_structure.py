"""Test the Structure class from clusterx.structure"""

import os

import pytest
from ase import Atoms
from ase.data import chemical_symbols
import numpy as np

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


def test_serialize_load(structure):
    filepath = "structure.json"
    structure.serialize(filepath=filepath, fmt=None)
    structure_loaded = Structure.from_file(filepath)
    assert structure == structure_loaded


def test_sigma_grid_pristine(super_cell):
    sigma_grid = np.array(
        [[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],]]],
        dtype=int
    )
    sigmas = sigma_grid.flatten()
    struct_flat = Structure(super_cell, sigmas=sigmas)
    struct_grid = Structure.from_sigma_grid(super_cell, sigma_grid)
    np.testing.assert_array_equal(
        struct_flat.get_sigmas(), struct_grid.get_sigmas())


def test_sigma_grid_orthorhombic(parent_lattice):
    p = [1, 2, 4]
    N = len(parent_lattice)
    grid_shape = p + [N]
    super_cell = SuperCell(parent_lattice, p)
    sites = super_cell.get_sites()
    sigmas = [len(site) - 1 for site in sites.values()]
    sigma_grid = np.array(sigmas).reshape(grid_shape)

    struct_flat = Structure(super_cell, sigmas=sigmas)
    struct_grid = Structure.from_sigma_grid(super_cell, sigma_grid)
    np.testing.assert_array_equal(
        struct_flat.get_sigmas(), struct_grid.get_sigmas())


def test_initializations(parent_lattice):
    p = [2, 2, 2]
    N = len(parent_lattice)
    grid_shape = p + [N]
    super_cell = SuperCell(parent_lattice, p)
    sites = super_cell.get_sites()
    sigmas = [len(site) - 1 for site in sites.values()]
    sigma_grid = np.array(sigmas).reshape(grid_shape)
    decoration = [site[-1] for site in sites.values()]
    decoration_symbols = [chemical_symbols[site[-1]] for site in sites.values()]

    reference = Structure(super_cell, sigmas=sigmas)
    others = []
    others.append(Structure(super_cell, decoration=decoration))
    others.append(Structure(super_cell, decoration_symbols=decoration_symbols))
    others.append(Structure.from_sigma_grid(super_cell, sigma_grid))

    for other in others:
        assert reference == other


def test_sigma_grid_non_diag(parent_lattice):
    p = [[1, 1, 0],
        [-1, 1, 0],
        [ 0, 0, 2]]
    super_cell = SuperCell(parent_lattice, p)
    with pytest.raises(ValueError):
        Structure.from_sigma_grid(super_cell, [])
