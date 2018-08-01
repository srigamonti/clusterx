import clusterx as c
from clusterx.utils import isclose
from clusterx.parent_lattice import ParentLattice
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.super_cell import SuperCell
from ase import Atoms
import numpy as np
import time

def test_clusters_generation():
    """Test generation of clusters pools.

    After successful execution of the test, the generated clusters may be visualized with the command::

        ase gui test_clusters_generation_#.json

    """
    tassert = True
    #######################################################
    # Part 1: 2D, radii.
    #######################################################
    """
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
        atom_idxs, atom_nrs = cp.get_cpool_arrays()
    """
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

    #######################################################
    # Part 3: FCC, radii.
    #######################################################
    print("Part III")

    from ase.build import bulk
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

    """
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
    """
    #######################################################
    # Part 5: Pt on O(111)
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
        if (mult != mult_ref).any() or not isclose(radii,radii_ref) or (npoints != npoints_ref).any():
            isok5 = False

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
        if (mult != mult_ref).any() or not isclose(radii,radii_ref) or (npoints != npoints_ref).any():
            isok6 = False


    print ("\n\n========Test writes========")
    print (test_clusters_generation.__doc__)
    print ("===========================\n")

    print ("========Asserts========")
    if tassert:
        print("test part 1")
        #assert isok1
        print("test part 2")
        assert isok2
        print("test part 3")
        assert isok3
        print("test part 4")
        #assert isok4
        print("test part 5")
        assert isok5
        print("test part 6")
        assert isok6
