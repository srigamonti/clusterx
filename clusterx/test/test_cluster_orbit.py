import numpy as np
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.symmetry import get_spacegroup
from ase import Atoms
from ase.db.jsondb import JSONDatabase
from subprocess import call
from ase.db.jsondb import JSONDatabase
import sys

def test_cluster_orbit():
    """Test creation of cluster orbit in supercell using spglib
    """

    a = 3.62/np.sqrt(2.0)
    positions = [(0,0,0)]
    cell = [(a,0,0),(0,a,0),(0,0,a)]
    pbc = (True,True,True)
    pri = Atoms('Cu', positions=positions, cell= cell, pbc= pbc)
    sub = Atoms('Al', positions=positions, cell= cell, pbc= pbc)

    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)
    scell = SuperCell(plat,[(5,0,0),(0,1,0),(0,0,1)])
    atnums = scell.get_atomic_numbers()
    sites = scell.get_sites()
    idx_subs = scell.get_idx_subs()
    tags = scell.get_tags()

    cl = ClustersPool(plat)

    orbit = cl.get_cluster_orbit(scell, [0])

    db_name = "test_orbit1.json"
    call(["rm","-f",db_name])
    atoms_db = JSONDatabase(filename=db_name) # For visualization
    for cl in orbit:
        atoms = scell.copy()
        ans = atnums
        for site in cl:
            ans[site] = sites[site][1]
        atoms.set_atomic_numbers(ans)
        atoms_db.write(atoms)
    
    print ("\n\n========Test writes========")
    print (test_cluster_orbit.__doc__)
    print(orbit)
    print ("===========================\n")

    print ("========Asserts========")
    
    #assert np.allclose(pl.get_scaled_positions(), correct_pos, atol=atol)

    
