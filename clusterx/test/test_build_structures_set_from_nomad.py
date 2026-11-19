import pytest

from clusterx.cli import build_structures_set_from_nomad as command_module
from clusterx.cli.build_structures_set_from_nomad import build_structures_set_from_nomad


class FakeParentLattice:
    def serialize(self, filepath):
        self.filepath = filepath


class FakeStructure:
    def __init__(self):
        self.parent_lattice = FakeParentLattice()

    def get_parent_lattice(self):
        return self.parent_lattice


class FakeDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def get_atoms_objects(self):
        self.calls.append("atoms")
        return ["relaxed atoms"]

    def get_total_energies(self):
        self.calls.append("energies")
        return [-1385771.1987808442]

    def get_band_gaps(self):
        self.calls.append("band_gaps")
        return [0.4]

    def get_structure_objects(self):
        self.calls.append("structures")
        return [FakeStructure()]

    def get_entry_ids(self):
        self.calls.append("entry_ids")
        return ["entry-id"]

    def get_lattice_constants(self):
        self.calls.append("lattice_constants")
        return [4.0], [5.0], [6.0], [90.0], [91.0], [92.0]

    def get_normalized_lattice_constants(self):
        self.calls.append("normalized_lattice_constants")
        return [3.8], [3.9], [11.7], [90.0], [90.0], [90.0]


class FakeStructuresSet:
    def __init__(self, parent_lattice):
        self.parent_lattice = parent_lattice
        self.properties = {}
        self.serialization = None

    def add_structures(self, structures):
        self.structures = structures

    def set_property_values(self, property_name, property_vals, update_db):
        assert update_db
        self.properties[property_name] = property_vals

    def serialize(self, **kwargs):
        self.serialization = kwargs

    def get_parent_lattice(self):
        return self.parent_lattice


def test_build_structures_set_from_nomad(monkeypatch):
    dataset_instances = []
    structures_sets = []
    writes = []

    def create_dataset(**kwargs):
        dataset = FakeDataset(**kwargs)
        dataset_instances.append(dataset)
        return dataset

    def create_sset(parent_lattice):
        sset = FakeStructuresSet(parent_lattice)
        structures_sets.append(sset)
        return sset

    monkeypatch.setattr(command_module, "get_nomad_oasis_access_token", lambda **kwargs: "token")
    monkeypatch.setattr(command_module, "Dataset", create_dataset)
    monkeypatch.setattr(command_module, "StructuresSet", create_sset)
    monkeypatch.setattr(command_module, "write", lambda **kwargs: writes.append(kwargs))

    result = build_structures_set_from_nomad(
        "dataset-id",
        base_url="https://example.org/api/v1/",
        username="user",
        userpassword="password",
        sset_filepath="output.db",
        atoms_db_filepath="atoms.db",
        save_atoms_db=True,
        parent_lattice_filepath="plat.json",
        save_parent_lattice=True,
        include_oasis_entry_id=True,
        include_total_energy_ev=True,
        include_band_gap_ev=True,
        include_lattice_params=True,
        include_normalized_lattice_params=True,
    )
    fake_sset = structures_sets[0]

    assert result is fake_sset
    assert dataset_instances[0].kwargs == {
        "dataset_id": "dataset-id",
        "token": "token",
        "pagination_page_size": 100,
        "base_url": "https://example.org/api/v1/",
    }
    assert dataset_instances[0].calls == [
        "structures",
        "atoms",
        "energies",
        "band_gaps",
        "entry_ids",
        "lattice_constants",
        "normalized_lattice_constants",
    ]
    assert writes == [{"filename": "atoms.db", "images": ["relaxed atoms"]}]
    assert fake_sset.parent_lattice is fake_sset.structures[0].get_parent_lattice()
    assert fake_sset.properties == {
        "oasis_entry_id": ["entry-id"],
        "total_energy_ev": [-1385771.1987808442],
        "band_gap_ev": [0.4],
        "a": [4.0],
        "b": [5.0],
        "c": [6.0],
        "alpha": [90.0],
        "beta": [91.0],
        "gamma": [92.0],
        "a_norm": [3.8],
        "b_norm": [3.9],
        "c_norm": [11.7],
        "alpha_norm": [90.0],
        "beta_norm": [90.0],
        "gamma_norm": [90.0],
    }
    assert fake_sset.serialization == {
        "filepath": "output.db",
        "ase_db_type": "db",
        "overwrite": True,
    }
    assert fake_sset.parent_lattice.filepath == "plat.json"


def test_build_structures_set_from_nomad_adds_no_properties_or_optional_files_by_default(monkeypatch):
    datasets = []
    structures_sets = []
    writes = []

    def create_dataset(**kwargs):
        dataset = FakeDataset(**kwargs)
        datasets.append(dataset)
        return dataset

    def create_sset(parent_lattice):
        sset = FakeStructuresSet(parent_lattice)
        structures_sets.append(sset)
        return sset

    monkeypatch.setattr(command_module, "get_nomad_oasis_access_token", lambda **kwargs: "token")
    monkeypatch.setattr(command_module, "Dataset", create_dataset)
    monkeypatch.setattr(command_module, "StructuresSet", create_sset)
    monkeypatch.setattr(command_module, "write", lambda **kwargs: writes.append(kwargs))

    result = build_structures_set_from_nomad("dataset-id")

    assert result is structures_sets[0]
    assert result.properties == {}
    assert datasets[0].calls == ["structures"]
    assert writes == []
    assert not hasattr(result.parent_lattice, "filepath")


def test_build_structures_set_from_nomad_rejects_empty_dataset(monkeypatch):
    class EmptyDataset(FakeDataset):
        def get_structure_objects(self):
            self.calls.append("structures")
            return []

    monkeypatch.setattr(command_module, "get_nomad_oasis_access_token", lambda **kwargs: "token")
    monkeypatch.setattr(command_module, "Dataset", EmptyDataset)
    monkeypatch.setattr(command_module, "write", lambda **kwargs: None)

    with pytest.raises(ValueError, match="did not contain any structures"):
        build_structures_set_from_nomad("empty-dataset")
