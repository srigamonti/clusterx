import numpy as np
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.symmetry import get_spacegroup
from ase import Atoms
from ase.spacegroup import crystal
from ase.build import bulk
import sys

def test_cluster_orbit():
    """Test creation of cluster orbit in supercell using spglib

    After running the test, the orbit can be visualized with the command::
        
        ase gui test_orbit1.json
    """
    #test_cases = [0,1,2,3]
    test_cases = [3]
    orbits = [None,None,None,None]
    for test_case in test_cases:
        if test_case == 0:
            # Perfect cubic lattice. The tested cluster is such that many interactions
            # with the periodic images of the crystal are present.
            a = 3.62/np.sqrt(2.0)
            positions = [(0,0,0)]
            cell = [(a,0,0),(0,a,0),(0,0,a)]
            pbc = (True,True,True)
            pri = Atoms('Cu', positions=positions, cell= cell, pbc= pbc)
            sub = Atoms('Al', positions=positions, cell= cell, pbc= pbc)
            sub2 = Atoms('Na', positions=positions, cell= cell, pbc= pbc)

            plat = ParentLattice(pri, substitutions=[sub,sub2], pbc=pbc)
            scell = SuperCell(plat,[(5,0,0),(0,2,0),(0,0,1)])

            cl = ClustersPool(plat)

            orbit = cl.get_cluster_orbit(scell, [0,2])
            db_name = "test_orbit%s.json"%(test_case)
            cl.write_orbit_db(orbit, scell, db_name)
            orbits[test_case] = orbit
            
        if test_case == 1:
            # FCC lattice
            pri = bulk('Cu', 'fcc', a=3.6)
            sub = bulk('Al', 'fcc', a=3.6)

            plat = ParentLattice(pri, substitutions=[sub], pbc=pri.get_pbc())
            scell = SuperCell(plat,[(2,0,0),(0,2,0),(0,0,2)])

            cl = ClustersPool(plat)

            orbit = cl.get_cluster_orbit(scell, [0,2])
            db_name = "test_orbit%s.json"%(test_case)
            cl.write_orbit_db(orbit, scell, db_name)
            orbits[test_case] = orbit
            
        if test_case == 2:
            # Clathrate 2x1x1 supercell. This contains spectator atoms.
            a = 10.515
            x = 0.185; y = 0.304; z = 0.116
            wyckoff = [
                (0, y, z), #24k
                (x, x, x), #16i
                (1/4., 0, 1/2.), #6c
                (1/4., 1/2., 0), #6d
                (0, 0 , 0) #2a
            ]

            pri = crystal(['Si','Si','Si','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a*1.0, a*1.0, a*1.0, 90, 90, 90])
            sub = crystal(['Al','Al','Al','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a*1.0, a*1.0, a*1.0, 90, 90, 90])

            plat = ParentLattice(atoms=pri,substitutions=[sub])
            scell = SuperCell(plat,[(2,0,0),(0,1,0),(0,0,1)])

            cl = ClustersPool(plat)
            orbit = cl.get_cluster_orbit(scell, [19,17]) # 24k-24k pair cluster
            db_name = "test_orbit%s.json"%(test_case)
            cl.write_orbit_db(orbit, scell, db_name)
            orbits[test_case] = orbit
            
        if test_case == 3:
            # Al(111) surface with Na substitution on the first layer. Test a 3-point cluster.
            from ase.build import fcc111, add_adsorbate
            pri = fcc111('Al', size=(1,1,3), vacuum=10.0)
            sub = pri.copy()
            for atom in sub:
                if atom.tag == 1:
                    atom.number = 11
        
            plat = ParentLattice(atoms=pri,substitutions=[sub])
            scell = SuperCell(plat,[(4,0,0),(0,4,0),(0,0,1)])

            cl = ClustersPool(plat)
            orbit = cl.get_cluster_orbit(scell, [2,14,5])
            db_name = "test_orbit%s.json"%(test_case)
            cl.write_orbit_db(orbit, scell, db_name)
            orbits[test_case] = orbit
            print(orbit)
            
    print ("\n\n========Test writes========")
    print (test_cluster_orbit.__doc__)
    #print(np.array2string(orbit1,separator=","))
    #print(np.array2string(orbit2,separator=","))
    #print(np.array2string(orbit3,separator=","))
    print ("===========================\n")

    print ("========Asserts========")

    for test_case in test_cases:
        assert check_result(test_case, orbits[test_case])

    
def check_result(testnr, orbit):
    isok = True
    if testnr == 0:
        rorbit = np.array([
            [0,2],
            [1,3],
            [2,4],
            [3,5],
            [4,6],
            [5,7],
            [6,8],
            [7,9],
            [8,0],
            [9,1],
            [0,1],
            [2,3],
            [4,5],
            [6,7],
            [8,9],
            [0,0],
            [1,1],
            [2,2],
            [3,3],
            [4,4],
            [5,5],
            [6,6],
            [7,7],
            [8,8],
            [9,9]])

    
    if testnr == 1:
        rorbit = np.array([
            [0,2],
            [1,3],
            [4,6],
            [5,7],
            [0,5],
            [1,4],
            [2,7],
            [3,6],
            [0,4],
            [1,5],
            [2,6],
            [3,7],
            [0,3],
            [1,2],
            [4,7],
            [5,6],
            [0,6],
            [1,7],
            [2,4],
            [3,5],
            [0,1],
            [2,3],
            [4,5],
            [6,7]])

    if testnr == 2:
        rorbit = np.array(
            [[19,17],
             [73,71],
             [70,72],
             [16,18],
             [ 4,60],
             [58, 6],
             [ 7,59],
             [61, 5],
             [12,14],
             [66,68],
             [69,67],
             [15,13],
             [62,64],
             [ 8,10],
             [65,63],
             [11, 9],
             [22,20],
             [76,74],
             [75,77],
             [21,23],
             [54,56],
             [ 0, 2],
             [ 3, 1],
             [57,55]] 
        )

    if testnr == 3:
        return True

    if len(orbit) != len(rorbit):
        return False
    
    for cl,rcl in zip(orbit,rorbit):
        if (cl != rcl).any():
            isok = False
            break
        
    return isok
    