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


def test_slicing_addition(structures_set):
    sset1 = structures_set[:5]
    sset2 = structures_set[5:]
    sset3 = sset1 + sset2
    assert isinstance(sset3, StructuresSet)
    assert len(sset3) == len(sset1) + len(sset2)


def test_property_transfer_addition(parent_lattice, super_cell):
    n_structures = 5
    sset1 = StructuresSet(parent_lattice)
    sset2 = StructuresSet(parent_lattice)
    sset3 = StructuresSet(parent_lattice)
    for i in range(n_structures):
        sset1.add_structure(super_cell.gen_random_structure())
        sset2.add_structure(super_cell.gen_random_structure())
        sset3.add_structure(super_cell.gen_random_structure())
    sset1.set_property_values(property_name="set_prop", property_vals=[1.0] * n_structures)
    sset3.set_property_values(property_name="set_prop", property_vals=[2.0] * n_structures)
    sset4 = sset1 + sset2 + sset3
    assert isinstance(sset4, StructuresSet)
    assert len(sset4) == len(sset1) + len(sset2) + len(sset3)
    np.testing.assert_array_equal(
        sset4.get_property_values("set_prop"), n_structures * [1.0] + n_structures * [None] + n_structures * [2.0]
    )


def test_serialize_load_json(structures_set):
    structures_set.set_property_values(property_name="set_prop", property_vals=[1.0] * len(structures_set))
    structures_set.serialize(filepath="sset.json", ase_db_type="json", overwrite=True, rm_vac=False)
    sset_loaded = StructuresSet(filepath="sset.json")
    assert len(sset_loaded) == len(structures_set)
    np.testing.assert_array_equal(sset_loaded.get_property_values("set_prop"), [1.0] * len(structures_set))


def test_serialize_load_sqlite(structures_set):
    structures_set.set_property_values(property_name="set_prop", property_vals=[1.0] * len(structures_set))
    structures_set.serialize(filepath="sset.db", ase_db_type="db", overwrite=True, rm_vac=False)
    sset_loaded = StructuresSet(filepath="sset.json")
    assert len(sset_loaded) == len(structures_set)
    np.testing.assert_array_equal(sset_loaded.get_property_values("set_prop"), [1.0] * len(structures_set))


def test_property_calculation_with_ase_calculator(structures_set):
    """Test calculation of properties with custom property solver ASE calculator"""
    structures_set.set_calculator(EMT())
    structures_set.compute_property_values(property_name="tote")

    tote_list = []
    for s in structures_set:
        ats = s.get_atoms().copy()
        ats.calc = EMT()
        tote_list.append(ats.get_potential_energy())

    np.testing.assert_array_equal(structures_set.get_property_values("tote"), tote_list)


def test_property_calculation_with_custom_solver(structures_set):
    """Test calculation of properties with custom property solver"""

    compute_ref_value = False

    def custom_prop(i, structure, **kwargs):
        par1 = kwargs["par1"]
        par2 = kwargs["par2"]
        e = (i * par1 - par2) * len(structure)
        return e

    structures_set.compute_property_values(property_name="cprop", property_calc=custom_prop, par1=3, par2=5)

    if compute_ref_value:
        custom_prop_list = []
        for i in range(len(structures_set)):
            custom_prop_list.append(custom_prop(i, structures_set[i], par1=3, par2=5))
        print(custom_prop_list)
    else:
        custom_prop_list = [-135, -54, 27, 108, 189, 270, 351, 432, 513, 594]

    np.testing.assert_array_equal(structures_set.get_property_values("cprop"), custom_prop_list)
