import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from ase import Atoms
import numpy as np

def test_cluster_correlations():
    """Test calculation of cluster correlations.
    
    After successful execution of the test, the generated structure and clusters pool may be visualized with the command::
        
        ase gui test_clusters_correlations_structure.json
        ase gui test_clusters_correlations_cpool.json

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

    plat = ParentLattice(pri,substitutions=[su1,su2,su3],pbc=pbc)
    cpool = ClustersPool(plat, npoints=[1,2], radii=[0,1.2])
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)

    scell = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    structure = Structure(scell,[1,1,1,6,7,1,1,2,1])
    corrs = corrcal.get_cluster_correlations(structure)
    
    # Generate output
    print ("\n\n========Test writes========")
    print (test_cluster_correlations.__doc__)
    atom_idxs, atom_nrs = cpool.get_cpool_arrays()
    scell = cpool.get_cpool_scell()
    cpool.write_orbit_db(cpool.get_cpool(),scell,"test_cluster_correlations_cpool.json")
    structure.serialize(fmt="json",fname="test_cluster_correlations_structure.json")
    print("Correlations: ",corrs)
    print ("===========================\n")

    print ("========Asserts========")
    
    assert np.allclose([-0.33333333,0.,-0.,0.33333333,0.57735027,-0.33333333,-0.25,-0.,-0.25],corrs,atol=1e-5)
    #print(corrs)
