# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell


@pytest.fixture
def parent_lattice():
    cu = bulk("Cu")
    return ParentLattice(atoms=cu, symbols=[["Cu", "Au"]])


@pytest.fixture
def super_cell(parent_lattice):
    return SuperCell(parent_lattice, 3)


@pytest.fixture
def structures_set_empty(parent_lattice):
    return StructuresSet(parent_lattice=parent_lattice)


@pytest.fixture
def structures_set(structures_set_empty, super_cell):
    np.random.seed(10)
    nstr1 = 10
    for i in range(nstr1):
        structures_set_empty.add_structure(super_cell.gen_random_structure())
    return structures_set_empty


def test_compute_property_values(structures_set):
    structures_set.set_calculator(EMT())
    structures_set.compute_property_values(property_name="tote")

    def a_prop(i, structure, **kwargs):
        at = structure.get_atoms()
        at.calc = EMT()
        return at.get_potential_energy() * 0.1 - 10

    structures_set.compute_property_values(property_name="a_prop0", property_calc=a_prop)
    structures_set.compute_property_values(property_name="a_prop1", property_calc=a_prop)


def test_set_property_values(structures_set):
    structures_set.set_property_values(property_name="set_prop", property_vals=[1.0] * len(structures_set))
    np.testing.assert_array_equal(structures_set.get_property_values("set_prop"), np.ones(len(structures_set)))


def test_indexing(structures_set):
    sset1 = structures_set[:5]
    sset2 = structures_set[5:]
    sset3 = sset1 + sset2
    assert isinstance(sset3, StructuresSet)
    assert len(sset3) == len(sset1) + len(sset2)


def test_add_sets_new(parent_lattice, super_cell):
    n_structures = 5
    sset1 = StructuresSet(parent_lattice)
    sset2 = StructuresSet(parent_lattice)
    for i in range(n_structures):
        sset1.add_structure(super_cell.gen_random_structure())
        sset2.add_structure(super_cell.gen_random_structure())
    sset1.set_property_values(property_name="set_prop", property_vals=[1.0] * n_structures)
    sset2.set_property_values(property_name="set_prop", property_vals=[2.0] * n_structures)
    sset3 = sset1 + sset2
    assert isinstance(sset3, StructuresSet)
    assert len(sset3) == len(sset1) + len(sset2)
    np.testing.assert_array_equal(sset3.get_property_values("set_prop"), n_structures * [1.0] + n_structures * [2.0])


def test_serialize_load(structures_set):
    structures_set.set_property_values(property_name="set_prop", property_vals=[1.0] * len(structures_set))
    structures_set.serialize(path="sset.json", overwrite=True, rm_vac=False)
    _ = StructuresSet(db_fname="sset.json")


def test_init(parent_lattice):
    """Test creation, union, serialization, and parsing of structures sets."""
    cu = bulk("Cu")
    plat = ParentLattice(atoms=cu, symbols=[["Cu", "Au"]])
    scell1 = SuperCell(plat, 3)

    sset1 = StructuresSet(parent_lattice=plat)
    np.random.seed(10)

    nstr1 = 10

    for i in range(nstr1):
        sset1.add_structure(scell1.gen_random_structure())

    sset1.set_calculator(EMT())
    sset1.compute_property_values(property_name="tote")

    def a_prop(i, structure, **kwargs):
        at = structure.get_atoms()
        at.calc = EMT()
        return at.get_potential_energy() * 0.1 - 10

    sset1.compute_property_values(property_name="a_prop0", property_calc=a_prop)
    sset1.compute_property_values(property_name="a_prop1", property_calc=a_prop)
    sset1.set_property_values(property_name="a_prop2", property_vals=[1.0] * len(sset1))

    sset1.serialize(path="sset1.json", overwrite=True)

    scell2 = SuperCell(plat, 2)
    nstr2 = 5
    sset2 = StructuresSet(parent_lattice=plat)

    sset2.compute_property_values(property_name="a_prop1", property_calc=a_prop)
    sset2.compute_property_values(property_name="a_prop2", property_calc=a_prop)

    for i in range(nstr2):
        sset2.add_structure(scell2.gen_random_structure())
    sset3 = sset1 + sset2
    assert len(sset3) == len(sset1) + len(sset2)
    sset3.serialize(path="sset3.json", overwrite=True, rm_vac=False)

    sset4 = StructuresSet(db_fname="sset3.json")
    sset4.serialize(path="sset4.json", overwrite=True, rm_vac=False)
