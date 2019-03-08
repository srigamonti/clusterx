# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet
import numpy as np
from ase.build import bulk
import sys


def test_structures_set():
    """Test creation, union, serialization, and parsing of structures sets.
    """

    cu = bulk("Cu")
    plat = ParentLattice(atoms=cu, symbols=[["Cu","Au"]])
    scell1 = SuperCell(plat, 3)

    sset1 = StructuresSet(parent_lattice=plat)
    np.random.seed(10)

    nstr1 = 10

    for i in range(nstr1):
        sset1.add_structure(scell1.gen_random())

    from ase.calculators.emt import EMT
    sset1.set_calculator(EMT())
    print(sset1.calculate_property(prop_name="tote"))

    def a_prop(structure):
        at = structure.get_atoms()
        at.set_calculator(EMT())
        return at.get_potential_energy()*0.1-10

    sset1.calculate_property(prop_name="a_prop0", prop_func=a_prop)
    sset1.calculate_property(prop_name="a_prop1", prop_func=a_prop)
    #sset1.calculate_property(prop_name="tote2")

    sset1.serialize(path="sset1.json", overwrite=True)

    scell2 = SuperCell(plat, 2)
    nstr2 = 5
    sset2 = StructuresSet(parent_lattice=plat)

    sset2.calculate_property(prop_name="a_prop1", prop_func=a_prop)
    sset2.calculate_property(prop_name="a_prop2", prop_func=a_prop)

    for i in range(nstr2):
        sset2.add_structure(scell2.gen_random())



    sset3 = sset1 + sset2

    sset3.serialize(path="sset3.json", overwrite=True, rm_vac=False)

    sset4 = StructuresSet(db_fname="sset3.json")
    sset4.serialize(path="sset4.json", overwrite=True, rm_vac=False)
