# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import clusterx as c
from clusterx.utils import isclose
from clusterx.parent_lattice import ParentLattice
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.super_cell import SuperCell
from ase import Atoms
import numpy as np
import time
from ase.build import bulk

def test_clusters_generation():
    """Test generation of clusters pools.

    After successful execution of the test, the generated clusters may be visualized with the command::

        ase gui test_clusters_generation_#.json

    """
    tassert = True
    doparts = [2,3,5,6,7]
    #doparts = [8]
    isok = []
    if 1 in doparts:
        #######################################################
        # Part 1: 2D, radii.
        #######################################################
        print("\nPart I")
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

        cp = ClustersPool(pl, npoints=[1,2,3], radii=[0,2.1,2.1])
        cp.write_clusters_db(db_name="test_clusters_generation_1.json")

        print("\nMult: ", cp.get_multiplicities())
        print("\nMult2: ", cp.get_cluster_multiplicities())

        if tassert:
            try:
                atom_idxs, atom_nrs = cp.get_cpool_arrays()
                isok.append(True)
            except:
                isok.append(False)

    if 2 in doparts:
        #######################################################
        # Part 2: 1D, supercell.
        #######################################################
        print("Part II")
        a=3.0
        cell = np.array([[1,0,0],[0,4,0],[0,0,1]])
        positions = np.array([[0,0,0],[0,1,0],[0,2,0],[0,3,0]])
        sites = [[12,13],[12,13],[12,13],[12,13]]
        pris = Atoms(cell=cell*a, positions=positions*a)

        pl = ParentLattice(pris, sites=sites, pbc=(1,0,0))
        sc = SuperCell(pl,[[4,0,0],[0,1,0],[0,0,1]])


        cp = ClustersPool(pl,npoints=[1,2],super_cell=sc) # Here radii are not given, therefore all the clusters which can fit in the supercell are generated.
        #cp = ClustersPool(pl,npoints=[0,1,2],radii=[0,0,3.9*a]) # Here radii are not given, therefore all the clusters which can fit in the supercell are generated.
        cp.write_clusters_db(db_name="test_clusters_generation_2.json")

        mult = cp.get_multiplicities()
        radii = cp.get_all_radii()
        npoints = cp.get_all_npoints()

        #atom_idxs, atom_nrs = cp.get_cpool_arrays()
        #print(atom_idxs[1][0],atom_idxs[7][0],atom_idxs[12][1],len(atom_nrs[17]),len(cp))

        if tassert:
            atom_idxs, atom_nrs = cp.get_cpool_arrays()
            isok2 = atom_idxs[1][0] == 1 and atom_idxs[7][0] == 2 and atom_idxs[12][1] == 8 and len(atom_nrs[17]) == 2 and len(cp) == 18
            isok.append(isok2)

    if 3 in doparts:
        #######################################################
        # Part 3: FCC, radii.
        #######################################################
        print("Part III")

        from ase.data import atomic_numbers as an
        a=4.1
        pris = bulk("Cu",crystalstructure="fcc", a=a)
        sites = [[an["Cu"],an["Au"]]]
        pl = ParentLattice(pris,sites=sites)

        cp = ClustersPool(pl,npoints=[0,1,2,3],radii=[0,0,5.0,5.0])
        cp.write_clusters_db(db_name="test_clusters_generation_3.json")

        mult = cp.get_multiplicities()
        npoints = cp.get_all_npoints()
        radii = cp.get_all_radii()

        rmult = np.array([1,1,6,3,8,12])
        rnpoints = np.array([0,1,2,2,3,3])
        rradii = np.array([0.       , 0.       , 2.8991378, 4.1      , 2.8991378, 4.1      ])

        if tassert:
            isok3 = len(cp) == 6 and (mult == rmult).all() and (npoints == rnpoints).all() and isclose(radii,rradii)
            isok.append(isok3)

    if 4 in doparts:
        #######################################################
        # Part 4: 2D cell with acute angle.
        #######################################################
        print("Part IV")
        from ase.io import write

        a=4.1
        cell = np.array([[5,0,0],[4,1,0],[0,0,1]])
        positions = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0]])
        pbc = (1,1,0)
        #sites = [[an["Cu"],an["Au"]]]*5
        sites = [[an["Cu"],an["Au"]],[an["Cu"],an["Au"]],[12],[12],[12]]
        pris = Atoms(cell=cell*a,positions=positions*a,pbc=pbc)
        pl = ParentLattice(pris,sites=sites,pbc=pbc)

        #cp = ClustersPool(pl,npoints=[0,1,2,3],radii=[0,0,4.5*a,4.5*a])
        cp = ClustersPool(pl,npoints=[2],radii=[2.0*a])
        write(filename="test_clusters_generation_scell_part4.json",images=cp.get_cpool_scell().get_atoms(),format="json")
        cp.write_clusters_db(db_name="test_clusters_generation_4.json")

        print("\nMult: ", cp.get_multiplicities())
        print("\nMult2: ", cp.get_cluster_multiplicities())
        if tassert:
            atom_idxs, atom_nrs = cp.get_cpool_arrays()
            isok4 = atom_idxs[2][0] == 30 and atom_idxs[3][1] == 16 and atom_idxs[4][0] == 10
            isok.append(isok4)

    if 5 in doparts:
        #######################################################
        # Part 5: O on Pt(111)
        #######################################################
        print("Part V")
        from ase.io import write

        cell = np.array([[2.785200119, 0.0, 0.0], [-1.3926000595, 2.4120540577, 0.0], [0.0, 0.0, 7.2740998268]])
        positions = np.array([[0.0, 0.0, 2.274101], [1.392600, 0.804018, 0.0], [0.0, 0.0, 2.916914], [1.392600, 0.804018, 2.924188], [0.0, 1.608036, 2.931462]])
        pbc = (1,1,0)
        sites = [[78], [78], [0,8], [0,8], [0,8]]
        pris = Atoms(cell=cell,positions=positions,pbc=pbc)
        pl = ParentLattice(pris,sites=sites,pbc=pbc)

        cp = ClustersPool(pl,npoints=[0,1,2,3],radii=[0,0,3.3,3.0])
        #cp = ClustersPool(pl,npoints=[0,1,2],radii=[0,0,3.3])
        #write(filename="test_clusters_generation_scell_part5.json",images=cp.get_cpool_scell().get_atoms(),format="json")
        cp.write_clusters_db(db_name="test_clusters_generation_5.json")

        mult = cp.get_multiplicities()
        radii = cp.get_all_radii()
        npoints = cp.get_all_npoints()

        #print(repr(mult))
        #print(repr(radii))
        #print(repr(npoints))

        mult_ref = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3, 3, 3, 1, 3, 1, 1, 1, 1, 1])
        radii_ref = np.array(
            [0.        , 0.        , 0.        , 0.        , 1.60805243,
            1.60805243, 1.60810181, 2.78520012, 2.78520012, 2.78520012,
            3.21608028, 3.21608028, 3.21610496, 1.60810181, 2.78520012,
            2.78520012, 2.78520012, 2.78520012, 2.78520012, 2.78520012,
            2.78520012, 2.78520012, 2.78520012, 2.78520012, 2.78520012,
            2.78520012])
        npoints_ref = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

        if tassert:
            isok5 = True
            if len(mult) != len(mult_ref) or len(npoints) != len(npoints_ref):
                isok5 = False
            elif (mult != mult_ref).any() or not isclose(radii,radii_ref) or (npoints != npoints_ref).any():
                isok5 = False
            isok.append(isok5)

    if 6 in doparts:
        #######################################################
        # Part 6: Clathrate
        #######################################################
        print("Part VI")
        from ase.io import write
        from ase.spacegroup import crystal
        a = 10.515
        #x = 0.185; y = 0.304; z = 0.116
        x = 0.1847; y = 0.2977; z = 0.1067
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

        #cp = ClustersPool(plat,npoints=[1,2],radii=[0,5.7],super_cell=SuperCell(plat,np.diag([2,1,1])))
        cp = ClustersPool(plat,npoints=[0,1,2],radii=[0,0,5.0])
        cp.write_clusters_db(db_name="test_clusters_generation_6.json")

        mult = cp.get_multiplicities()
        radii = cp.get_all_radii()
        npoints = cp.get_all_npoints()

        mult_ref = np.array([1, 24, 16, 6, 12, 8, 48, 24, 48, 24, 48, 48, 48, 12, 24, 24])
        radii_ref = np.array([0.0,0.0,0.0,0.0,2.243901, 2.378554359,2.419983103,2.606790868,3.817351239, 3.88424099, 3.88473654, 3.90320808, 3.979588005, 4.254368999, 4.258122441, 4.31192162])
        npoints_ref = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        if tassert:
            isok6 = True
            if len(mult) != len(mult_ref) or len(npoints) != len(npoints_ref):
                isok6 = False
            elif (mult != mult_ref).any() or not isclose(radii,radii_ref) or (npoints != npoints_ref).any():
                isok6 = False
            isok.append(isok6)

    if 7 in doparts:
        #######################################################
        # Part 7: Negative radii
        #######################################################

        print("Part VII")

        plat = ParentLattice(
            Atoms(cell=np.diag([2,2,5]),positions=[[0,0,0]]),
            site_symbols=[["Cu","Al"]],
            pbc=(1,1,0)
            )

        scell = SuperCell(plat,np.array([(6,0,0),(0,6,0),(0,0,1)]))
        cp = ClustersPool(plat, npoints=[0,1,2,3,4], radii=[0,0,-1,4.1,2.9], super_cell=scell)

        cp.write_clusters_db(db_name="test_clusters_generation_7.json")

        mult = cp.get_multiplicities()
        radii = cp.get_all_radii()
        npoints = cp.get_all_npoints()

        print(repr(mult))
        print(repr(radii))
        print(repr(npoints))

        mult_ref = np.array([1, 1, 2, 2, 2, 4, 2, 2, 4, 4, 2, 4, 2, 4, 1])
        radii_ref = np.array([0.        , 0.        , 2.        , 2.82842712, 4.        ,
           4.47213595, 5.65685425, 6.        , 6.32455532, 7.21110255,
           8.48528137, 2.82842712, 4.        , 4.        , 2.82842712])
        npoints_ref = np.array([0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4])

        if tassert:
            isok7 = True
            if len(mult) != len(mult_ref) or len(npoints) != len(npoints_ref):
                isok7 = False
            elif (mult != mult_ref).any() or not isclose(radii,radii_ref) or (npoints != npoints_ref).any():
                isok7 = False
            isok.append(isok7)

    if 8 in doparts:
        #######################################################
        # Part 8: FCC(111) with alloying and Adsorption in hollow sites
        #######################################################

        print("Part VIII")
        from ase.build import fcc111, add_adsorbate

        pristine = fcc111('Re', size=(1,1,3), a=3.2) # 3-atomic-layer Al slab
        add_adsorbate(pristine,'X',1.7,position='fcc') # Hollow FCC vacancy site
        pristine.center(vacuum=10.0, axis=2) # add vacuum along z-axis

        symbols = [['Co'],['Co'],['Co','Ni'],['X','Al']]
        platt = ParentLattice(pristine, symbols=symbols)

        scell = SuperCell(platt,[[5,0],[0,2]])
        scell.serialize(fname="scell.json")
        print(scell.get_sublattice_types())

        npoints = [2]
        radii = [-1]
        cp = ClustersPool(platt, npoints=npoints, radii=radii, super_cell=scell,method=1)

        cp.write_clusters_db(db_name="test_clusters_generation_8.json")

        mult = cp.get_multiplicities()
        radii = cp.get_all_radii()
        npoints = cp.get_all_npoints()

        #print(repr(mult))
        #print(repr(radii))
        #print(repr(npoints))

        mult_ref = np.array([3, 3, 3, 2, 3, 3, 2, 2, 5])
        radii_ref = np.array([2.14398383, 2.2627417 , 2.2627417 , 3.11715682, 3.91918359,
       3.91918359, 4.5254834 , 4.5254834 , 5.0076608 ])
        npoints_ref = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])

        if tassert:
            isok8 = True
            if len(mult) != len(mult_ref) or len(npoints) != len(npoints_ref):
                isok8 = False
            elif (mult != mult_ref).any() or not isclose(radii,radii_ref) or (npoints != npoints_ref).any():
                isok8 = False
            isok.append(isok8)

    print ("\n\n========Test writes========")
    print (test_clusters_generation.__doc__)
    print ("===========================\n")

    print ("========Asserts========")
    if tassert:
        for i,p in enumerate(doparts):
            print("Test part ",p)
            assert isok[i]
        """
        #print("test part 1")
        #assert isok1
        print("test part 2")
        assert isok2
        print("test part 3")
        assert isok3
        #print("test part 4")
        #assert isok4
        print("test part 5")
        assert isok5
        print("test part 6")
        assert isok6
        print("test part 7")
        assert isok7
        """
