# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit
# https://www.apache.org/licenses/LICENSE-2.0.txt.

from ase import Atoms
from ase.data import atomic_numbers as an
from ase.io import write
import numpy as np
from ase.build import bulk, fcc111, add_adsorbate

from clusterx.utils import isclose
from clusterx.parent_lattice import ParentLattice
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.super_cell import SuperCell
from clusterx.test.defaults import get_clathrate_plat

# TODO: fix reference/expected values
def test_2D_radii():
    cell = [
        [3,0,0],
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

    pl = ParentLattice(pri, substitutions=[su1,su2,su3], pbc=pbc)
    cp = ClustersPool(pl, npoints=[1,2,3], radii=[0,2.1,2.1])
    cp.write_clusters_db(db_name="test_clusters_generation_1.json")
    cp.get_multiplicities()
    atom_idxs, atom_nrs = cp.get_cpool_arrays()


def test_2D_supercell():
    a=3.0
    cell = np.array([
        [1,0,0],
        [0,4,0],
        [0,0,1]])
    positions = np.array([
        [0,0,0],
        [0,1,0],
        [0,2,0],
        [0,3,0]])
    sites = [[12,13],[12,13],[12,13],[12,13]]
    pri = Atoms(cell=cell*a, positions=positions*a)

    pl = ParentLattice(pri, sites=sites, pbc=(1,0,0))
    sc = SuperCell(pl,[[4,0,0],[0,1,0],[0,0,1]])
    # Here radii are not given, therefore all the clusters which can fit in the
    # supercell are generated.
    cp = ClustersPool(pl,npoints=[1,2],super_cell=sc) 
    cp.write_clusters_db(db_name="test_clusters_generation_2.json")

    atom_idxs, atom_nrs = cp.get_cpool_arrays()
    assert atom_idxs[1][0] == 1
    assert atom_idxs[7][0] == 2
    assert atom_idxs[12][1] == 8
    assert len(atom_nrs[17]) == 2
    assert len(cp) == 18


def test_fcc_radii():
    a=4.1
    pris = bulk("Cu", crystalstructure="fcc", a=a)
    sites = [[an["Cu"], an["Au"]]]
    pl = ParentLattice(pris,sites=sites)

    cp = ClustersPool(pl, npoints=[0,1,2,3], radii=[0,0,5.0,5.0])
    cp.write_clusters_db(db_name="test_clusters_generation_3.json")

    mult = cp.get_multiplicities()
    npoints = cp.get_all_npoints()
    radii = cp.get_all_radii()

    rmult = np.array([1,1,6,3,8,12])
    rnpoints = np.array([0,1,2,2,3,3])
    rradii = np.array([0., 0., 2.8991378, 4.1, 2.8991378, 4.1])

    assert len(cp) == 6 
    assert (mult == rmult).all() 
    assert (npoints == rnpoints).all()
    assert isclose(radii,rradii)


def test_2D_acute_angle():
    a=4.1
    cell = np.array([
        [5,0,0],
        [4,1,0],
        [0,0,1]])
    positions = np.array([
        [0,0,0],
        [1,0,0],
        [2,0,0],
        [3,0,0],
        [4,0,0]])
    pbc = (1,1,0)
    #sites = [[an["Cu"],an["Au"]]]*5
    sites = [[an["Cu"],an["Au"]],[an["Cu"],an["Au"]],[12],[12],[12]]
    pris = Atoms(cell=cell*a,positions=positions*a,pbc=pbc)
    pl = ParentLattice(pris,sites=sites,pbc=pbc)

    #cp = ClustersPool(pl,npoints=[0,1,2,3],radii=[0,0,4.5*a,4.5*a])
    cp = ClustersPool(pl,npoints=[2],radii=[2.0*a])
    atoms = cp.get_cpool_scell().get_pristine_structure().get_atoms()
    write(
        filename="test_clusters_generation_scell_part4.json",
        images=atoms,
        format="json")
    cp.write_clusters_db(db_name="test_clusters_generation_4.json")
    cp.get_multiplicities()

    atom_idxs, atom_nrs = cp.get_cpool_arrays()
    assert atom_idxs[2][0] == 30
    assert atom_idxs[3][1] == 16
    assert atom_idxs[4][0] == 10


def test_pt_111():
    cell = np.array([
        [2.785200119  , 0.0         , 0.0         ],
        [-1.3926000595, 2.4120540577, 0.0         ],
        [0.0          , 0.0         , 7.2740998268]])
    positions = np.array([
        [0.0     , 0.0     , 2.274101],
        [1.392600, 0.804018, 0.0     ],
        [0.0     , 0.0     , 2.916914],
        [1.392600, 0.804018, 2.924188],
        [0.0     , 1.608036, 2.931462]])
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

    mult_ref = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3, 3,
        3, 1, 3, 1, 1, 1, 1, 1])
    radii_ref = np.array(
        [0.        , 0.        , 0.        , 0.       , 1.60805243,
        1.60805243, 1.60810181, 2.78520012, 2.78520012, 2.78520012,
        3.21608028, 3.21608028, 3.21610496, 1.60810181, 2.78520012,
        2.78520012, 2.78520012, 2.78520012, 2.78520012, 2.78520012,
        2.78520012, 2.78520012, 2.78520012, 2.78520012, 2.78520012,
        2.78520012])
    npoints_ref = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3])

    assert len(mult) == len(mult_ref) 
    assert len(npoints) == len(npoints_ref)
    assert (mult == mult_ref).all()
    assert isclose(radii,radii_ref)
    assert (npoints == npoints_ref).all()


def test_clathrate():
    plat = get_clathrate_plat()

    #cp = ClustersPool(plat,npoints=[1,2],radii=[0,5.7],super_cell=SuperCell(plat,np.diag([2,1,1])))
    cp = ClustersPool(plat,npoints=[0,1,2],radii=[0,0,5.0])
    cp.write_clusters_db(db_name="test_clusters_generation_6.json")

    mult = cp.get_multiplicities()
    radii = cp.get_all_radii()
    npoints = cp.get_all_npoints()

    mult_ref = np.array(
        [1, 24, 16, 6, 12, 8, 48, 24, 48, 24, 48, 48, 48, 12, 24, 24])
    radii_ref = np.array(
        [0.0, 0.0, 0.0, 0.0, 2.243901, 2.378554359,2.419983103,2.606790868,
        3.817351239, 3.88424099, 3.88473654, 3.90320808, 3.979588005,
        4.254368999, 4.258122441, 4.31192162])
    npoints_ref = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    assert len(mult) == len(mult_ref)
    assert len(npoints) == len(npoints_ref)
    assert (mult == mult_ref).all()
    assert isclose(radii, radii_ref)
    assert (npoints == npoints_ref).all()


def test_negative_radii():
    plat = ParentLattice(
        Atoms(cell=np.diag([2,2,5]),positions=[[0,0,0]]),
        site_symbols=[["Cu","Al"]],
        pbc=(1,1,0)
    )

    scell = SuperCell(plat,np.array([(6,0,0),(0,6,0),(0,0,1)]))
    cp = ClustersPool(
        plat, npoints=[0,1,2,3,4], radii=[0,0,-1,4.1,2.9], super_cell=scell)
    cp.write_clusters_db(db_name="test_clusters_generation_7.json")

    mult = cp.get_multiplicities()
    radii = cp.get_all_radii()
    npoints = cp.get_all_npoints()

    mult_ref = np.array([1, 1, 2, 2, 2, 4, 2, 2, 4, 4, 2, 4, 2, 4, 1])
    radii_ref = np.array([
        0.        , 0.        , 2.        , 2.82842712, 4.        ,
        4.47213595, 5.65685425, 6.        , 6.32455532, 7.21110255,
        8.48528137, 2.82842712, 4.        , 4.        , 2.82842712])
    npoints_ref = np.array([0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4])

    assert len(mult) == len(mult_ref)
    assert len(npoints) == len(npoints_ref)
    assert (mult == mult_ref).all()
    assert isclose(radii, radii_ref)
    assert (npoints == npoints_ref).all()


def test_fcc_111():
    """FCC(111) with alloying and Adsorption in hollow sites"""
    pristine = fcc111('Re', size=(1,1,3), a=3.2) # 3-atomic-layer Al slab
    add_adsorbate(pristine,'X',1.7,position='fcc') # Hollow FCC vacancy site
    pristine.center(vacuum=10.0, axis=2) # add vacuum along z-axis

    symbols = [['Co'],['Co'],['Co','Ni'],['X','Al']]
    platt = ParentLattice(pristine, symbols=symbols)

    scell = SuperCell(platt,[[5,0],[0,2]])
    scell.serialize(fname="scell.json")
    scell.get_sublattice_types()

    npoints = [2]
    radii = [-1]
    cp = ClustersPool(
        platt, npoints=npoints, radii=radii, super_cell=scell,method=1)
    cp.write_clusters_db(db_name="test_clusters_generation_8.json")

    mult = cp.get_multiplicities()
    radii = cp.get_all_radii()
    npoints = cp.get_all_npoints()

    mult_ref = np.array([3, 3, 3, 2, 3, 3, 2, 2, 5])
    radii_ref = np.array(
        [2.14398383, 2.2627417 , 2.2627417 ,
         3.11715682, 3.91918359, 3.91918359,
         4.5254834 , 4.5254834 , 5.0076608])
    npoints_ref = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])

    assert len(mult) == len(mult_ref)
    assert len(npoints) == len(npoints_ref)
    assert (mult == mult_ref).all()
    assert isclose(radii, radii_ref)
    assert (npoints == npoints_ref).all()
