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
    sset = StructuresSet(plat)

    nst = 15
    for i in range(nst):
        sset.add_structure(scell.gen_random(nsubs={0:[4]}))

    isok1 = True
    isok2 = True
    isok3 = True

    sset.write_files(prefix = "random_strs-")
    sset.calculate_energies(EMT2())
    sset.read_property_values()

    """
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
