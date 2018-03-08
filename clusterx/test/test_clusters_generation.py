import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.clusters.clusters_pool import ClustersPool
from ase import Atoms
import subprocess


def test_clusters_generation():
    """Test generation of clusters pools.
    
    After successful execution of the test, the generated clusters may be visualized with the command::
        
        ase gui test_clusters_generation_#.json

    """

    # Clusters pool with corrdump
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

    cp = ClustersPool(pl, npoints=[2], radii=[2.5], tool="corrdump")
    cp.gen_clusters()
    #print(cp.get_clusters_dict())
    cp.serialize("atat")
    cp.serialize("json",fname = "test_clusters_generation_0.json")
    
    #subprocess.call(["rm","-f","parlat.in"])
    #subprocess.call(["rm","-f","clusters.out"])
