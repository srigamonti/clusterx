# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from ase import Atoms
from ase.spacegroup import crystal
from ase.build import bulk
import sys

def test_cluster_orbit2():
    """Test creation of cluster orbit in supercell using spglib

    After running the test, the orbit can be visualized with the command::

        ase gui test_cluster_orbit_#.json
    """
    tassert = True
    #tassert = False
    test_cases = [0,1,2,3,4,5]
    #test_cases = [0]
    orbits = [None,None,None,None,None,None]
    mults = [None,None,None,None,None,None]
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
            scell = SuperCell(plat,[(5,0,0),(0,2,0),(0,0,1)], sym_table = True)

            cl = ClustersPool(plat)

            orbit,mult = cl.get_cluster_orbit2(scell, [0,2], [11,11])
            db_name = "test_cluster_orbit_%s.json"%(test_case)
            cl.write_clusters_db(orbit, scell, db_name)
            orbits[test_case] = orbit

        if test_case == 1:
            # FCC lattice
            pri = bulk('Cu', 'fcc', a=3.6)
            sub = bulk('Al', 'fcc', a=3.6)

            plat = ParentLattice(pri, substitutions=[sub], pbc=pri.get_pbc())
            scell = SuperCell(plat,np.diag([2,2,2]), sym_table = True)

            cl = ClustersPool(plat)

            orbit,mult = cl.get_cluster_orbit2(scell, [0,2],[13,13])
            db_name = "test_cluster_orbit_%s.json"%(test_case)
            cl.write_clusters_db(orbit, scell, db_name)
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

            pri = crystal(['Si','Si','Si','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
            sub = crystal(['Al','Al','Al','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])

            plat = ParentLattice(atoms=pri,substitutions=[sub])
            scell = SuperCell(plat,[(2,0,0),(0,1,0),(0,0,1)], sym_table = True)

            cl = ClustersPool(plat)
            orbit,mult = cl.get_cluster_orbit2(scell, [19,17],[13,13]) # 24k-24k pair cluster
            db_name = "test_cluster_orbit_%s.json"%(test_case)
            cl.write_clusters_db(orbit, scell, db_name)
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
            scell = SuperCell(plat,[(4,0,0),(0,4,0),(0,0,1)], sym_table = True)

            cl = ClustersPool(plat)
            orbit,mult = cl.get_cluster_orbit2(scell, [2,14,5],[11,11,11])
            db_name = "test_cluster_orbit_%s.json"%(test_case)
            cl.write_clusters_db(orbit, scell, db_name)
            orbits[test_case] = orbit

        if test_case == 4:
            # Al(111) surface with Na substitution on the first layer and on-top Oxygen adsorption.
            from ase.build import fcc111, add_adsorbate
            from ase.visualize import view

            pri = fcc111('Al', size=(1,1,3))
            add_adsorbate(pri,'X',1.5,'ontop')
            pri.center(vacuum=10.0, axis=2)

            sub1 = pri.copy() # Na substitution on the first Al layer
            for atom in sub1:
                if atom.tag == 1:
                    atom.number = 11

            sub2 = pri.copy() # O on-top adsorbates
            for atom in sub2:
                if atom.tag == 0:
                    atom.number = 8

            plat = ParentLattice(atoms=pri,substitutions=[sub1,sub2])
            scell = SuperCell(plat,[(4,0,0),(0,4,0),(0,0,1)], sym_table = True)

            cl = ClustersPool(plat)
            orbit,mult = cl.get_cluster_orbit2(scell, [3,18],[8,11])
            db_name = "test_cluster_orbit_%s.json"%(test_case)
            cl.write_clusters_db(orbit, scell, db_name)
            orbits[test_case] = orbit

        if test_case == 5:
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
            scell = SuperCell(plat,[(5,0,0),(0,2,0),(0,0,1)], sym_table = True)
            sites = scell.get_sites()
            cl = ClustersPool(plat)

            atom_idxs = [0,2]
            atom_species = [sites[0][1],sites[2][2]]
            orbit,mult = cl.get_cluster_orbit2(scell, atom_idxs, atom_species)
            db_name = "test_cluster_orbit_%s.json"%(test_case)
            cl.write_clusters_db(orbit, scell, db_name)
            orbits[test_case] = orbit

        
        orbits[test_case] = orbit
        mults[test_case] = mult

        gen_ref = False
        if gen_ref:
            print("********++++++++   test_case = "+str(test_case)+"   ++++++++*********")
            _orbit_nrs = []
            _orbit_idxs = []
            for i in range(len(orbit)):
                _orbit_nrs.append(orbit[i].get_nrs())
                _orbit_idxs.append(orbit[i].get_idxs())
            print(np.array2string(np.array(_orbit_idxs),separator=","))
            print(mult)

    print ("\n\n========Test writes========")
    print (test_cluster_orbit2.__doc__)
    #print(np.array2string(orbit1,separator=","))
    print ("===========================\n")

    print ("========Asserts========")

    if tassert:
        for test_case in test_cases:
            print("test orbit: ",test_case)
            assert check_result(test_case, orbits[test_case])


def check_result(testnr, orbit):
    isok = True
    orbit_nrs = []
    orbit_idxs = []
    for i in range(0,len(orbit)):
        orbit_nrs.append(orbit[i].get_nrs())
        orbit_idxs.append(orbit[i].get_idxs())
    
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
        rorbit = np.array([[0,2],
                           [1,3],
                           [4,6],
                           [5,7],
                           [0,4],
                           [1,5],
                           [2,6],
                           [3,7],
                           [0,5],
                           [1,4],
                           [2,7],
                           [3,6],
                           [0,3],
                           [1,2],
                           [4,7],
                           [5,6],
                           [0,1],
                           [2,3],
                           [4,5],
                           [6,7],
                           [0,6],
                           [1,7],
                           [2,4],
                           [3,5]])

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
        rorbit = np.array(
            [[ 2, 14,  5],
             [ 5, 17,  8],
             [ 8, 20, 11],
             [11, 23,  2],
             [14, 26, 17],
             [17, 29, 20],
             [20, 32, 23],
             [23, 35, 14],
             [26, 38, 29],
             [29, 41, 32],
             [32, 44, 35],
             [35, 47, 26],
             [38,  2, 41],
             [41,  5, 44],
             [44,  8, 47],
             [47, 11, 38]])

    if testnr == 4:
        rorbit = np.array(
            [[ 3, 18],
             [ 7, 22],
             [11, 26],
             [15, 30],
             [19, 34],
             [23, 38],
             [27, 42],
             [31, 46],
             [35, 50],
             [39, 54],
             [43, 58],
             [47, 62],
             [51,  2],
             [55,  6],
             [59, 10],
             [63, 14],
             [ 3, 54],
             [ 7, 58],
             [11, 62],
             [15, 50],
             [19,  6],
             [23, 10],
             [27, 14],
             [31,  2],
             [35, 22],
             [39, 26],
             [43, 30],
             [47, 18],
             [51, 38],
             [55, 42],
             [59, 46],
             [63, 34],
             [ 3, 14],
             [ 7,  2],
             [11,  6],
             [15, 10],
             [19, 30],
             [23, 18],
             [27, 22],
             [31, 26],
             [35, 46],
             [39, 34],
             [43, 38],
             [47, 42],
             [51, 62],
             [55, 50],
             [59, 54],
             [63, 58],
             [ 3, 30],
             [ 7, 18],
             [11, 22],
             [15, 26],
             [19, 46],
             [23, 34],
             [27, 38],
             [31, 42],
             [35, 62],
             [39, 50],
             [43, 54],
             [47, 58],
             [51, 14],
             [55,  2],
             [59,  6],
             [63, 10],
             [ 3, 50],
             [ 7, 54],
             [11, 58],
             [15, 62],
             [19,  2],
             [23,  6],
             [27, 10],
             [31, 14],
             [35, 18],
             [39, 22],
             [43, 26],
             [47, 30],
             [51, 34],
             [55, 38],
             [59, 42],
             [63, 46],
             [ 3,  6],
             [ 7, 10],
             [11, 14],
             [15,  2],
             [19, 22],
             [23, 26],
             [27, 30],
             [31, 18],
             [35, 38],
             [39, 42],
             [43, 46],
             [47, 34],
             [51, 54],
             [55, 58],
             [59, 62],
             [63, 50]])

    if testnr == 5:
        return True

    if len(orbit_idxs) != len(rorbit):
        return False
    
    #sorted_rorbit = [sorted(x) for x in rorbit]
    
    #for cl in sorted(orbit_idxs, key=lambda x: x[0]):
    #    if cl not in sorted_rorbit:
    #        isok = False
    #        break
        
    for cl,rcl in zip(orbit_idxs,rorbit):
        if (cl != np.sort(rcl)).any():
            return False

    #for cl,rcl in zip(orbit_idxs,rorbit):
    #    print(cl,rcl)
    #    if (cl != np.sort(rcl)).any():
    #        isok = False
    #        break
    return isok


#if __name__ == "__main__":
#    test_cluster_orbit()
