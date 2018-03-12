import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.clusters.clusters_pool import ClustersPool
from ase import Atoms


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
    #cp.write_orbit_db(atom_idxs,scell,"test_clusters_generation_0.json",orbit_species=atom_nrs)
    cp.write_orbit_db(cp.get_cpool(),scell,"test_clusters_generation_0.json",orbit_species=atom_nrs)
