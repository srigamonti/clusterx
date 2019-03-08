# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet
from clusterx.calculators.emt import EMT2
#from ase.calculators.emt import EMT
import numpy as np
from ase import Atoms

def test_folders():
    """
    Test creation of folders for ab-initio runs and reading of property values
    from those folders.
    """

    a = 3.0
    plat = ParentLattice(Atoms(cell=a*np.diag([1,1,1]), positions=[[0,0,0]],numbers=[29]),site_symbols=[["Cu","Au"]],pbc=(1,1,1))
    scell = SuperCell(plat, np.diag([2,2,2]))
    sset1 = StructuresSet(plat)

    nst = 15
    for i in range(nst):
        sset1.add_structure(scell.gen_random(nsubs={0:[4]}))

    isok1 = True
    isok2 = True
    isok3 = True

    sset1.write_files(prefix = "random_strs-") # tests folder creation and writing of structure files
    sset1.calculate_energies(EMT2()) # test iterating over folders and calculating energies and writing energy files
    sset1.read_property_values(property_name = "my_total_energy") # test
    e1 = sset1.get_property_values("my_total_energy")

    sset2 = StructuresSet(plat)
    sset2.add_structures(structures=sset1.get_db_fname()) # Test adding structures from json database file
    for s1,s2 in zip(sset1,sset2):
        print((s1.get_atomic_numbers()==s2.get_atomic_numbers()).all())

    #sset3 = StructuresSet(plat, db_fname=sset1.get_db_fname()) # Test init structures set from folders database
    sset3 = StructuresSet(db_fname=sset1.get_db_fname()) # Test init structures set from folders database
    e3 = sset3.get_property_values("my_total_energy")

    print((np.array(e1)==np.array(e3)).all())



    """
    for s1,s2 in zip(sset1,sset2):
        print(s1,s2)
        print(s1.get_atomic_numbers(),s2.get_atomic_numbers())

    try:
        sset.write_files(prefix = "random_strs-")
    except:
        isok1 = False

    try:
        sset.calculate_energies(EMT2())
    except:
        isok2 = False

    try:
        sset.read_property_values()
    except:
        isok3 = False
    """

    assert isok1
    assert isok2
    assert isok3
