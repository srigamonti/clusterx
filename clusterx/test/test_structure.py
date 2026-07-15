"""Test the Structure class from clusterx.structure"""

import pytest
from ase import Atoms
from ase.data import chemical_symbols
import numpy as np

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure


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
    return SuperCell(parent_lattice, [(1, 0, 0), (0, 3, 0), (0, 0, 1)])


@pytest.fixture
def structure(super_cell):
    return Structure(super_cell, decoration=[1] * len(super_cell))


@pytest.fixture
def structure_random(super_cell: SuperCell):
    # nsubs = {0: [3], 1: [1, 1, 1], 2: [2, 1]}  # failing for some reason
    nsubs = None
    return super_cell.gen_random_structure(nsubs=nsubs, track_indices=True)


def test_get_atoms(structure):
    atoms = structure.get_atoms()
    assert isinstance(atoms, Atoms)
    numbers = structure.get_atomic_numbers()
    assert len(numbers) == 9
    symbols = structure.get_chemical_symbols()
    assert len(symbols) == 9
    atoms = structure.get_atoms(ignore_observer_atoms=True)
    assert len(atoms) == 6


def test_random_flips(structure_random: Structure):
    for _ in range(100):
        i, si, sj, site_type = structure_random.get_random_flip()
        structure_random.flip(i, si, sj, site_type)


def test_random_swaps(structure_random: Structure):
    for _ in range(10):
        try:
            i, j, site_type = structure_random.get_random_swap(site_types=None)
        except RuntimeError:
            print("Impossible substitution for swap!")
            continue
        structure_random.swap(i, j, site_type)


def test_large_index_map(parent_lattice):
    p = 200
    super_cell = SuperCell(parent_lattice, [p, p, 1])
    structure = Structure(super_cell, decoration=[1] * len(super_cell), track_indices=False)
    assert not structure._track_indices
    structure.track_indices()
    assert len(structure._index_maps) == 3


def test_reduced_structure(parent_lattice):
    p = [10, 10, 10]
    p_reduced = [3, 3, 3]
    super_cell = SuperCell(parent_lattice, p)
    super_cell_reduced = SuperCell(parent_lattice, p_reduced)
    sites = super_cell.get_sites()
    sigmas = [len(site) - 1 for site in sites.values()]
    structure = Structure(super_cell, sigmas=sigmas)
    structure.init_reduced_structure(super_cell_reduced)
    structure_reduced, index_reduced = structure.get_reduced_structure(0)
    assert isinstance(structure_reduced, Structure)
    assert isinstance(index_reduced, int)


def test_init_scell(structure):
    assert structure.scell is not None


@pytest.mark.parametrize(
    "p",
    [
        [(1, 0, 0), (0, 3, 0), (0, 0, 1)],
        [3, 3, 1],
        [100, 100, 1],
    ],
)
def test_serialize_load(parent_lattice, p, tmp_path):
    # ensure no bottleneck with large supercell size
    super_cell = SuperCell(parent_lattice, p)
    structure = Structure(super_cell, decoration=[1] * len(super_cell))
    filepath = tmp_path / "structure.json"
    structure.serialize(filepath=filepath, fmt=None)
    structure_loaded = Structure.from_file(filepath)
    assert structure == structure_loaded


def test_sigma_grid_pristine(super_cell):
    sigma_grid = np.array(
        [
            [
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ]
        ],
        dtype=int,
    )
    sigmas = sigma_grid.flatten()
    struct_flat = Structure(super_cell, sigmas=sigmas)
    struct_grid = Structure.from_sigma_grid(super_cell, sigma_grid)
    np.testing.assert_array_equal(struct_flat.get_sigmas(), struct_grid.get_sigmas())


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
    np.testing.assert_array_equal(struct_flat.get_sigmas(), struct_grid.get_sigmas())
    np.testing.assert_array_equal(struct_flat.sigmas, struct_grid.sigmas)


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
        np.testing.assert_array_equal(reference.sigmas, other.sigmas)


def test_swap(super_cell):
    sigmas_init = [0, 1, 0, 0, 2, 0, 0, 0, 0]
    sigmas_final = [0, 2, 0, 0, 1, 0, 0, 0, 0]
    i, j = 1, 4
    structure_init = Structure(super_cell, sigmas=sigmas_init)
    structure_init.swap(i, j)
    structure_final = Structure(super_cell, sigmas=sigmas_final)
    assert structure_init == structure_final


def test_sigma_grid_non_diag(parent_lattice):
    p = [[1, 1, 0], [-1, 1, 0], [0, 0, 2]]
    super_cell = SuperCell(parent_lattice, p)
    with pytest.raises(ValueError):
        Structure.from_sigma_grid(super_cell, [])
