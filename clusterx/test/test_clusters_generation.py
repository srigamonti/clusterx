import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.clusters.clusters_pool import ClustersPool
from ase import Atoms
import numpy as np

def test_clusters_generation():
    """Test generation of clusters pools.
    
    After successful execution of the test, the generated clusters may be visualized with the command::
        
        ase gui test_clusters_generation_#.json

    """

    cell = [[3,0,0],
            [0,1,0],
            [0,0,5]]
    positions = [
        [0,0,0],
        [1,0,0],
        [2,0,0]]
    pbc = [True,True,False]

    pri = Atoms(['H','H','H'], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(['C','H','H'], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(['H','He','H'], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(['H','N','H'], positions=positions, cell=cell, pbc=pbc)

    pl = ParentLattice(pri,substitutions=[su1,su2,su3],pbc=pbc)

    #cp = ClustersPool(pl, npoints=[1,2,3,4,5], radii=[0,3.1,3.1,3.1,3.1])
    #cp = ClustersPool(pl, npoints=[1,2,3,4], radii=[0,3.1,3.1,3.1])
    cp = ClustersPool(pl, npoints=[1,2,3], radii=[0,3.1,1.5])
    atom_idxs, atom_nrs = cp.get_cpool_arrays()
    scell = cp.get_cpool_scell()
    cp.write_orbit_db(cp.get_cpool(),scell,"test_clusters_generation_0.json")


    print ("\n\n========Test writes========")
    print (test_clusters_generation.__doc__)
    #print(np.array2string(atom_idxs,separator=","))
    #print(np.array2string(atom_nrs,separator=","))

    print ("===========================\n")

    print ("========Asserts========")
    ratom_idxs = [(0,),(1,),(1,),(0, 1),(0, 1),(0, 3),(0, 4),(0, 4),(0, 6),(0, 7),(0, 7),
                  (0, 9),(0, 12),(1, 4),(1, 4),(1, 4),(1, 7),(1, 7),(1, 7),(1, 10),(1, 10),
                  (1, 10),(1, 12),(1, 12),(1, 13),(1, 13),(1, 13),(1, 15),(1, 15),(1, 18),
                  (1, 18),(0, 1, 3),(0, 1, 3),(0, 1, 4),(0, 1, 4),(0, 1, 4),(0, 1, 4)]

    ratom_nrs = [(6,),(2,),(7,),(6, 2),(6, 7),(6, 6),(6, 2),(6, 7),(6, 6),(6, 2),(6, 7),
                 (6, 6),(6, 6),(2, 2),(2, 7),(7, 7),(2, 2),(2, 7),(7, 7),(2, 2),(2, 7),
                 (7, 7),(2, 6),(7, 6),(2, 2),(2, 7),(7, 7),(2, 6),(7, 6),(2, 6),(7, 6),
                 (6, 2, 6),(6, 7, 6),(6, 2, 2),(6, 2, 7),(6, 7, 2),(6, 7, 7)]


    isok = True
    for ai,rai in zip(atom_idxs,ratom_idxs):
        if not np.allclose(ai,rai):
            isok = False
    assert isok
    
    isok = True
    for an,ran in zip(atom_nrs,ratom_nrs):
        if not np.allclose(an,ran):
            isok = False
    assert isok
