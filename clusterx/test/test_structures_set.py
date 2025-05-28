# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet


def test_structures_set():
    """Test creation, union, serialization, and parsing of structures sets."""
    cu = bulk("Cu")
    plat = ParentLattice(atoms=cu, symbols=[["Cu","Au"]])
    scell1 = SuperCell(plat, 3)

    sset1 = StructuresSet(parent_lattice=plat)
    np.random.seed(10)

    nstr1 = 10

    for i in range(nstr1):
        sset1.add_structure(scell1.gen_random_structure())
    
    sset1.set_calculator(EMT())
    sset1.compute_property_values(prop_name="tote")

    def a_prop(i, structure, **kwargs):
        at = structure.get_atoms()
        at.calc = EMT()
        return at.get_potential_energy()*0.1-10

    sset1.compute_property_values(prop_name="a_prop0", property_calc=a_prop)
    sset1.compute_property_values(prop_name="a_prop1", property_calc=a_prop)

    sset1.serialize(path="sset1.json", overwrite=True)

    scell2 = SuperCell(plat, 2)
    nstr2 = 5
    sset2 = StructuresSet(parent_lattice=plat)

    sset2.compute_property_values(prop_name="a_prop1", property_calc=a_prop)
    sset2.compute_property_values(prop_name="a_prop2", property_calc=a_prop)

    for i in range(nstr2):
        sset2.add_structure(scell2.gen_random_structure())
    sset3 = sset1 + sset2
    assert len(sset3) == len(sset1) + len(sset2)
    sset3.serialize(path="sset3.json", overwrite=True, rm_vac=False)

    sset4 = StructuresSet(db_fname="sset3.json")
    sset4.serialize(path="sset4.json", overwrite=True, rm_vac=False)
