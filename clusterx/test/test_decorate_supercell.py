import numpy as np
import pytest
from ase import Atoms

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.utils import decorate_supercell


def _make_two_site_scell(site_symbols):
    cell = np.diag([2.0, 2.0, 2.0])
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    atoms = Atoms(["H", "H"], positions=positions, cell=cell, pbc=(1, 1, 1))
    plat = ParentLattice(atoms=atoms, site_symbols=site_symbols, pbc=atoms.get_pbc())
    return SuperCell(plat, np.eye(3, dtype=int))


def test_decorate_supercell_infers_missing_vacancy():
    scell = _make_two_site_scell([["H"], ["H", "X", "He"]])
    atoms = Atoms(
        ["H"],
        positions=[[0.0, 0.0, 0.0]],
        cell=scell.get_cell(),
        pbc=scell.get_pbc(),
    )

    structure = decorate_supercell(scell, atoms, tol=1e-8)

    np.testing.assert_array_equal(structure.get_atomic_numbers(), [1, 0])


def test_decorate_supercell_rejects_missing_atom_without_vacancy_sublattice():
    scell = _make_two_site_scell([["H"], ["H", "He"]])
    atoms = Atoms(
        ["H"],
        positions=[[0.0, 0.0, 0.0]],
        cell=scell.get_cell(),
        pbc=scell.get_pbc(),
    )

    with pytest.raises(ValueError, match="no substitutional sublattice allowing vacancies"):
        decorate_supercell(scell, atoms, tol=1e-8)


def test_decorate_supercell_checks_expected_vacancy_count():
    scell = _make_two_site_scell([["H"], ["H", "X", "He"]])
    atoms = Atoms(
        ["H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=scell.get_cell(),
        pbc=scell.get_pbc(),
    )

    with pytest.raises(ValueError, match="expected 0 from atom counts"):
        decorate_supercell(scell, atoms, tol=1e-8)
