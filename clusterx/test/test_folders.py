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
    sset1.read_property_values() # test

    sset2 = StructuresSet(plat)
    sset2.add_structures(json_db_filepath=sset1.get_folders_db_fname()) # Test adding structures from json database file
    for s1,s2 in zip(sset1,sset2):
        print((s1.get_atomic_numbers()==s2.get_atomic_numbers()).all())

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
