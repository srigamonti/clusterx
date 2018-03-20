import clusterx as c
import subprocess
from ase.spacegroup import crystal
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.clusters.cluster import Cluster
from ase import Atoms

def test_clathrate_mc():

    subprocess.call(["rm","-f","test_clathrate_mc-cluster_orbit.json"])
    subprocess.call(["rm","-f","test_clathrate_mc-cpool.json"])
    
    a = 10.5148
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

    for i,p in enumerate(pri.get_scaled_positions()):
        print(i,p)


        
    plat = ParentLattice(atoms=pri,substitutions=[sub])
    scell = SuperCell(plat,[(1,0,0),(0,1,0),(0,0,1)])
    cpool = ClustersPool(plat, npoints=[0,1,2], radii=[0,0,4.4])
    #cpool_array = []

    #cpool_array.append(Cluster())

    print(plat.get_idx_subs())
    
    #cl = ClustersPool(plat)
    cpool.write_orbit_db(cpool.get_cpool(),cpool.get_cpool_scell(),"test_clathrate_mc-cpool.json")

    #orbit = cpool.get_cluster_orbit(scell, [19,17],[13,13]) # 24k-24k pair cluster
    #db_name = "test_clathrate_mc-cluster_orbit.json"
    #cl.write_orbit_db(orbit, scell, db_name)
    
