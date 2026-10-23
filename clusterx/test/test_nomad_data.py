import numpy as np
import pytest
from ase import Atoms

from clusterx.dbinterfaces.get_nomad_data import Dataset


class FakeSuperCell:
    def get_transformation(self):
        return np.eye(3)


class FakeStructure:
    def get_supercell(self):
        return FakeSuperCell()


def test_get_normalized_lattice_constants_returns_lengths_and_angles(monkeypatch):
    cell = np.array(
        [
            [0.0, 4.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 4.0],
        ]
    )
    dataset = Dataset(dataset_id="dataset", pagination_page_size=10)
    monkeypatch.setattr(dataset, "get_atoms_objects", lambda: [Atoms(cell=cell)])
    monkeypatch.setattr(dataset, "get_structure_objects", lambda: [FakeStructure()])

    a, b, c, alpha, beta, gamma = dataset.get_normalized_lattice_constants()

    assert a == pytest.approx([2.0])
    assert b == pytest.approx([4.0])
    assert c == pytest.approx([np.sqrt(18.0)])
    assert alpha == pytest.approx([np.degrees(np.arccos(4.0 / np.sqrt(18.0 * 16.0)))])
    assert beta == pytest.approx([np.degrees(np.arccos(2.0 / (2.0 * np.sqrt(18.0))))])
    assert gamma == pytest.approx([90.0])


def test_get_lattice_constants_preserves_cell_axis_order_without_normalization(monkeypatch):
    cell = np.diag([4.0, 2.0, 3.0])
    dataset = Dataset(dataset_id="dataset", pagination_page_size=10)
    monkeypatch.setattr(dataset, "get_atoms_objects", lambda: [Atoms(cell=cell)])

    a, b, c, alpha, beta, gamma = dataset.get_lattice_constants()

    assert a == pytest.approx([4.0])
    assert b == pytest.approx([2.0])
    assert c == pytest.approx([3.0])
    assert alpha == pytest.approx([90.0])
    assert beta == pytest.approx([90.0])
    assert gamma == pytest.approx([90.0])
