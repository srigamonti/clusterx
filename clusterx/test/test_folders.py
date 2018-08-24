from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet
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
        sset.add_structure(scell.gen_random(nsubs={0:[4]}), write_to_db = True)

    sset.write_files(root = ".", prefix = "test_folders-", suffix = "_tmp", fnames=["struc.json","geometry.xml","geometry.in"], formats = ["json","exciting","aims"])
